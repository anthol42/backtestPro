import numpy as np

from .trade import Trade, BuyLong, BuyShort, SellLong, SellShort, TradeType
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Optional
from datetime import datetime, timedelta


class Position:
    """
    Data class holding info about a position
    """

    def __init__(self, ticker: str, amount: int, long: bool, average_price: float, average_filled_time: datetime,
                 ratio_owned: float = 1.0):
        self.ticker = ticker
        self.amount = amount
        self.average_price = average_price
        self.on_margin = ratio_owned < 1.0
        self.long = long
        if amount < 0:
            raise ValueError("Amount cannot be negative nor zero")
        if not long and ratio_owned != 0.:
            raise ValueError("Short position cannot own shares.  Set ratio_owned to 0.")
        self.average_filled_time = average_filled_time
        self.last_dividends_dt = average_filled_time
        # Ratio of the security held in account that are owned by the investor.
        # Example:
        # - I bought 50 shares and used margin to buy another 50 shares.  I have a total of 100 shares, but my \
        #   ratio_owned is 0.5 because I only own 50 shares.
        self.ratio_owned = ratio_owned
        # This index correspond to the amount of shares hold times the time hold.  It will be used to calculate the
        # dividends due to the shareholders.
        # Formula: days * (amount)
        self.time_stock_idx = 0
        self._number_of_entry = 1

    @property
    def purchase_worth(self):
        return self.average_price * self.amount

    def __str__(self):
        mrg = "MARGIN" if self.on_margin else ""
        lg = "LONG" if self.long else "SHORT"
        return f"POSITION: {self.amount}x{self.ticker} {round(self.average_price, 2)}$ {mrg} {lg}"

    def __repr__(self):
        return f"POSITION: {self.ticker}"

    def update_time_stock_idx(self, timestep_elapsed: int = 1):
        """
        Call this method to update the time_stock_idx.  It will be used to calculate the dividends due to the shareholders.
        Call this method at the end of each day.
        :param timestep_elapsed: The number of seconds elapsed since the last call to this method
        :return: None
        """
        if self.long:
            self.time_stock_idx += timestep_elapsed * self.amount

    def dividends_got_paid(self, timestamp: datetime):
        """
        Call this method to reset last dividends date.  Useful when calculating how much dividends the user should have
        :param timestamp: The timestamp of dividend payment
        :return: None
        """
        self.last_dividends_dt = timestamp
        self.time_stock_idx = 0

    def __add__(self, other: Union['Position', Trade]):
        """
        Add a position to the current one.  It will update the average price and the amount, the amount borrowed, the
        average filled time, the last dividends date and the time_stock_idx.
        (for a better calculation of dividends due to shareholders)
        Note:
            If adding a trade that doesn't have the same own ratio, the ratio will be updated.
        :param other: The other position or trade to add.
        :return: None
        :raise TypeError: If trying to add long and short positions together
        :raise ValueError: If trying to add positions of different tickers or wrong trade type
        :raise RuntimeError: If trying to add a trade that happened before the last dividends [Not causal]
        :raise NotImplementedError: If trying to add a type that is not implemented
        """
        if isinstance(other, Position):
            if self.long != other.long:
                raise TypeError("Cannot add long and short positions together")
            if self.ticker != other.ticker:
                raise ValueError("Cannot add positions of different tickers")
            new = deepcopy(self)
            if self.amount == 0:
                # Initiate a new position without changing time_stock_idx in case the position pay dividends.
                new.amount = other.amount
                new.ratio_owned = other.ratio_owned
                new.average_price = other.average_price
                new.average_filled_time = other.average_filled_time
                new._number_of_entry = 1
                new.last_dividends_dt = other.last_dividends_dt
                new.on_margin = new.ratio_owned < 1.0
                return new
            # Concatenating positions
            current_amount = self.amount
            other_amount = other.amount
            total_amount = current_amount + other_amount
            new.average_price = ((current_amount / total_amount) * self.average_price +
                                 (other_amount / total_amount) * other.average_price)
            new.amount += other.amount
            new.time_stock_idx += other.time_stock_idx
            new.average_filled_time = (self.average_filled_time +
                                       timedelta(seconds=(other.average_filled_time -
                                                          self.average_filled_time).total_seconds() / 2))

            new.last_dividends_dt = (self.last_dividends_dt + timedelta(seconds=(other.last_dividends_dt -
                                                                                 self.last_dividends_dt).total_seconds() / 2))
            new._number_of_entry = self._number_of_entry + other._number_of_entry
            return new

        elif isinstance(other, Trade):
            if self.ticker != other.security:
                raise ValueError("Cannot add positions of different tickers")
            if self.long:
                if other.trade_type == TradeType.SellLong:
                    raise ValueError("Cannot add a trade that is selling to long position")
                elif other.trade_type != TradeType.BuyLong:
                    raise ValueError("Invalid trade type for addition to long position")
            else:  # Short
                if other.trade_type == TradeType.BuyShort:
                    raise ValueError("Cannot add a trade that is buying to short position")
                elif other.trade_type != TradeType.SellShort:
                    raise ValueError("Invalid trade type for addition to short position")

            new = deepcopy(self)
            if self.amount == 0:
                # Initiate a new position without changing time_stock_idx in case the position pay dividends.
                new.amount = other.amount + other.amount_borrowed
                new.ratio_owned = other.amount / new.amount
                new.average_price = other.security_price
                new.average_filled_time = other.timestamp
                new._number_of_entry = 1
                new.last_dividends_dt = other.timestamp
                new.on_margin = new.ratio_owned < 1.0
                return new

            # Newly acquired position.  (Was not held before)
            current_amount = self.amount
            other_amount = other.amount + other.amount_borrowed
            total_amount = current_amount + other_amount
            new.average_price = ((current_amount / total_amount) * self.average_price +
                                 (other_amount / total_amount) * other.security_price)
            new.amount += other.amount + other.amount_borrowed
            amount_owned = self.amount * self.ratio_owned + other.amount
            new.ratio_owned = amount_owned / new.amount
            # time stock idx is not updated when adding a trade because the position was not held before.
            new.time_stock_idx += 0
            new.average_filled_time = (self.average_filled_time +
                                       timedelta(seconds=(other.timestamp -
                                                          self.average_filled_time).total_seconds() / 2))
            new._number_of_entry += 1
            if other.timestamp < self.last_dividends_dt:
                raise RuntimeError("Cannot add a trade that happened before the last dividends [Not causal]")
            new.last_dividends_dt = (self.last_dividends_dt + timedelta(seconds=(other.timestamp -
                                                                                 self.last_dividends_dt).total_seconds() / 2))
            return new
        else:
            raise NotImplementedError(f"Addition not implemented for type {type(other)}")

    def __sub__(self, other):
        """
        Subtract a position from the current one.  It will update the average price and the amount and the amount
        borrowed.  The amount borrowed doesn't change because it is assumed to be done
        uniformly.
        :param other: The other position to add
        :return: None
        :raise TypeError: If trying to subtract long and short positions together
        :raise ValueError: If trying to subtract positions of different tickers or wrong trade type
        :raise NotImplementedError: If trying to subtract a type that is not implemented
        """
        if isinstance(other, Position):
            if self.long != other.long:
                raise TypeError("Cannot add long and short positions together")
            if self.ticker != other.ticker:
                raise ValueError("Cannot add positions of different tickers")
            new = deepcopy(self)
            new.amount -= other.amount
            return new
        elif isinstance(other, Trade):
            if self.ticker != other.security:
                raise ValueError("Cannot add positions of different tickers")
            if self.long:
                if other.trade_type == TradeType.BuyLong:
                    raise ValueError("Cannot sub a buying trade to a long position")
                elif other.trade_type != TradeType.SellLong:
                    raise ValueError("Invalid trade type for subtraction to a long position")
            else:  # Short
                if other.trade_type == TradeType.SellShort:
                    raise ValueError("Cannot sub a selling trade to a short position")
                elif other.trade_type != TradeType.BuyShort:
                    raise ValueError("Invalid trade type for subtraction to short position")
            new = deepcopy(self)
            new.amount -= other.amount + other.amount_borrowed
            return new
        else:
            raise NotImplementedError(f"Addition not implemented for type {type(other)}")

    def export(self) -> dict:
        """
        Export the position to a dictionary JSONable
        :return: The object as a dictionary
        """
        return {
            "type": "Position",
            "ticker": self.ticker,
            "amount": float(self.amount),
            "long": bool(self.long),
            "average_price": float(self.average_price),
            "on_margin": bool(self.on_margin),
            "average_filled_time": str(self.average_filled_time),
            "last_dividends_dt": str(self.last_dividends_dt),
            "time_stock_idx": self.time_stock_idx,
            "ratio_owned": float(self.ratio_owned),
            "number_of_entry": int(self._number_of_entry)
        }

    @classmethod
    def load(cls, data: dict):
        """
        Load the object from a dictionary
        :param data: The dictionary to load from
        :return: The object
        """
        self = cls(data["ticker"], data["amount"], data["long"], data["average_price"],
                   datetime.fromisoformat(data["average_filled_time"]), ratio_owned=data["ratio_owned"])
        self.last_dividends_dt = datetime.fromisoformat(data["last_dividends_dt"])
        self.time_stock_idx = data["time_stock_idx"]
        self._number_of_entry = data["number_of_entry"]
        return self

    def __eq__(self, other):
        return (self.ticker == other.ticker and
                self.amount == other.amount and
                self.long == other.long and
                self.average_price == other.average_price and
                self.on_margin == other.on_margin and
                self.average_filled_time == other.average_filled_time and
                self.last_dividends_dt == other.last_dividends_dt and
                self.time_stock_idx == other.time_stock_idx)


class TradeStats:
    """
    This class will hold the stats for a given trade.  It will be used to calculate the stats for the whole portfolio
    """

    def __init__(self, trade: Trade, duration: timedelta, profit: float, rel_profit: float, ratio_owned):
        """
        :param trade: The trade object (SellLong or BuyShort)
        :param duration: The duration of the trade
        :param profit: The profit made on the trade
        :param rel_profit: The profit made on the trade relative to the amount invested NOT IN PERCENTAGE
        :param ratio_owned: The ratio of the security held in account that are owned by the investor.  (not bought on
                            margin)
        :raise ValueError: If the trade type is not SellLong or BuyShort (Doesn't exit a position)
        """
        if trade.trade_type != TradeType.SellLong and trade.trade_type != TradeType.BuyShort:
            raise ValueError(f"Invalid trade type for TradeStats.  The tradetype must exit a position.  "
                             f"Got {trade.trade_type}, but expected SellLong or BuyShort.")
        self.trade = trade
        self.duration = duration
        self.profit = profit
        self.rel_profit = rel_profit
        self.ratio_owned = ratio_owned

    def export(self):
        """
        Export the object to a JSONable dictionary
        Note: The duration is saved in seconds
        :return: A dictionary corresponding top the object's state
        """
        return {
            "type": "TradeStats",
            "trade": self.trade.export(),
            "duration": self.duration.total_seconds(),
            "profit": self.profit,
            "rel_profit": self.rel_profit,
            "ratio_owned": self.ratio_owned
        }

    @classmethod
    def load(cls, data: dict):
        """
        Load the object from a dictionary
        :param data: The dictionary to load from
        :return: The object
        """
        trade = Trade.load(data["trade"])
        duration = timedelta(seconds=data["duration"])
        return cls(trade, duration, data["profit"], data["rel_profit"], data["ratio_owned"])

    def __eq__(self, other):
        return (self.trade == other.trade and
                self.duration == other.duration and
                self.profit == other.profit and
                self.rel_profit == other.rel_profit)


class Portfolio:
    """
    This  class will have two sub portfolios: long and short.  When 'trade' is called, it will add or remove in the
    appropriate portfolio the security.  It will also remember each trades (in a list) to recover the state at each
    days (For debugging purpose).
    """

    def __init__(self, transaction_cost: float = 10., transaction_relative: bool = False,
                 debt_record: Optional[Dict[str, float]] = None):
        """
        :param transaction_cost: The cost of a transaction (buy or sell) in $ or in %
        :param transaction_relative: Whether the transaction cost is in percentage relative to transaction cost or fix price
        :param debt_record: The amount of debt used to buy securities: {security: amount}.  Passed by reference from broker.
        """
        # Keys will be security (ticker) and the value will be Equity data object
        self._long: Dict[str, Position] = {}
        self._short: Dict[str, Position] = {}
        self._trades: List[Union[TradeStats, Trade]] = []  # TradeStats when closing trade and Trade when opening one.
        if transaction_relative and transaction_cost > 1:
            raise ValueError("Transaction cost cannot be greater than 1 when relative")
        if transaction_relative:
            self._transaction_cost = transaction_cost + 1
        else:
            self._transaction_cost = transaction_cost
        self._relative = transaction_relative
        self._debt_record: Dict[str, float] = debt_record if debt_record is not None else {}
        self._transaction_ids = set()
        self._short_len = 0
        self._long_len = 0

    def trade(self, trade: Trade) -> float:
        """
        Make a trade and add it to the portfolio.
        Note:
            This class will handle the debt record.  It will add the amount borrowed to the debt record when buying long
            and remove it when selling long.
        :param trade: Can be BuyLong, SellLong, SellShort, BuyShort
        :return: The cash change in account.  If negative, withdraw.  If positive, deposit.
        :raise RuntimeError: If the transaction ID is already used
        """
        if trade.transaction_id in self._transaction_ids:
            raise RuntimeError("Transaction ID already used.")
        else:
            self._transaction_ids.add(trade.transaction_id)

        if trade.trade_type == TradeType.BuyLong:
            """
            What this section is doing:
                1. Add the trade to the long portfolio (2 case: New position or adding to an existing one)
                2. Add the amount borrowed to the debt record (2 case: relative transaction cost or absolute)
                3. Save the trade for stats
                4. Compute the worth of the trade
            """
            if trade.security in self._long:
                self._long[trade.security] += trade
            else:
                self._long[trade.security] = Position(trade.security, trade.amount + trade.amount_borrowed, True,
                                                      trade.security_price, trade.timestamp,
                                                      trade.amount / (trade.amount + trade.amount_borrowed))
                self._long_len += 1

            # Handle debt record
            if self._relative:
                newdebt = trade.amount_borrowed * trade.security_price * self._transaction_cost
            else:
                # If the transaction cost is not relative, it is handled in the cost of the trade
                newdebt = trade.amount_borrowed * trade.security_price

            if trade.security in self._debt_record:
                self._debt_record[trade.security] += newdebt
            else:
                self._debt_record[trade.security] = newdebt

            # Save for stats
            self._trades.append(trade)

            return -self._getCost(trade, include_borrow=False)
        elif trade.trade_type == TradeType.SellLong:
            """
            What this section is doing:
                1. Verify that we have a long open position in the portfolio for this security. (Raise RuntimeError if not)
                2. Verify that we have enough shares to execute the trade. (Raise RuntimeError if not)
                3. Compute trade stats.
                4. Pay relative debt (Relative to the amount of shares sold compared to the total amount of shares)
                5. Update portfolio.
                6. Compute the worth of the trade (Market value - transaction - relative debt repaid) (2 case: relative 
                    transaction cost or absolute)
            """
            if trade.security not in self._long:
                raise RuntimeError("Cannot sell Long if the security is not acquired.")
            elif self._long[trade.security].amount < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot sell Long more securities than the portfolio has")
            else:
                # Get relative debt (Debt that will be repaid to the bank).
                trade_assets = trade.amount + trade.amount_borrowed
                total_assets = self._long[trade.security].amount
                relative_debt = self._debt_record[trade.security] * (trade_assets / total_assets)

                # Compute stats
                duration = trade.timestamp - self._long[trade.security].average_filled_time
                average_buy_price = self._long[trade.security].average_price
                ratio_owned = self._long[trade.security].ratio_owned
                number_of_entry = self._long[trade.security]._number_of_entry
                profit, rel_profit = self.getLongProfit(average_buy_price,
                                                        trade.security_price,
                                                        trade.amount + trade.amount_borrowed,
                                                        ratio_owned,
                                                        relative_debt, number_of_entry)
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               profit, rel_profit, ratio_owned))

                # Handle debt record
                self._debt_record[trade.security] -= relative_debt  # Debt that has been repaid
                self._long[trade.security] -= trade
                # Since we included their cost in the exit, if there are still shares in the position, we won't
                # include the transaction entry cost in the calculation of the profit for the next trade.
                self._long[trade.security]._number_of_entry = 0
                if self._long[trade.security].amount == 0:
                    self._long_len -= 1

                return self._getCost(trade, include_borrow=True, sell=True) - relative_debt

        elif trade.trade_type == TradeType.SellShort:
            """
            What this section is doing:
                1. Add the trade to the long portfolio (2 case: New position or adding to an existing one)
                2. Save the trade for stats
                3. Compute the worth of the trade
            """
            if trade.security in self._short:
                self._short[trade.security] += trade
            else:
                self._short[trade.security] = Position(trade.security, trade.amount_borrowed, False,
                                                       trade.security_price, trade.timestamp, 0)
                self._short_len += 1
            self._trades.append(trade)
            return self._getCost(trade, include_borrow=True, sell=True)
        elif trade.trade_type == TradeType.BuyShort:
            """
            What this section is doing:
                1. Verify that we have a short open position in the portfolio for this security. (Raise RuntimeError if not)
                2. Verify that we have enough shares to execute the trade. (Raise RuntimeError if not)
                3. Compute trade stats.
                4. Update portfolio.
                5. Compute the worth of the trade
            """
            if trade.security not in self._short:
                raise RuntimeError("Cannot buy short if the security has not been sold short.")
            elif self._short[trade.security].amount < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot buy short more securities than the portfolio has sold short.")
            else:
                # Compute stats
                duration = trade.timestamp - self._short[trade.security].average_filled_time
                average_sell_price = self._short[trade.security].average_price
                absolute_profit, rel_profit = self.getShortProfit(average_sell_price, trade.security_price, trade.amount_borrowed + trade.amount)
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               absolute_profit,
                               rel_profit,
                               0.))

                # Update portfolio
                self._short[trade.security] -= trade
                if self._short[trade.security].amount == 0:
                    self._short_len -= 1

                # Compute trade value
                return -self._getCost(trade, include_borrow=True)

    @property
    def len_long(self):
        return self._long_len
    @property
    def len_short(self):
        return self._short_len

    def getShortProfit(self, average_sell_price: float, average_buy_price: float, qty: int) -> Tuple[float, float]:
        """
        Calculate the profit made on a short trade
        :param average_sell_price: The price at which the security was sold
        :param average_buy_price: The price at which the security was bought
        :param qty: The number of shares in the trade
        :return: The profit (positive) or loss (negative) made on the trade, relative profit
        """
        intial_investment = average_sell_price * qty
        if self._relative:
            gain = (intial_investment * (2 - self._transaction_cost)) - (average_buy_price * qty * self._transaction_cost)
            return gain, 100 * gain / intial_investment
        else:
            gain = (intial_investment - self._transaction_cost) - (average_buy_price * qty + self._transaction_cost)
            return gain, 100 * gain / intial_investment

    def getLongProfit(self, average_buy_price: float, average_sell_price: float, qty: int, ratio_owned: float,
                      debt: float, number_of_entry: int) -> Tuple[float, float]:
        """
        Calculate the profit made on a long trade
        :param average_buy_price: The price at which the security was bought
        :param average_sell_price: The price at which the security was sold
        :param qty: The number of shares in the trade
        :param ratio_owned: The ratio of the security held in account that are owned by the investor.  (not bought on
                            margin)
        :param debt: The amount of debt used to buy the security
        :param number_of_entry: The number of time the position was entered.  (Used only if transaction cost is absolute)
        :return: The profit (positive) or loss (negative) made on the trade, relative profit in percentage
        """
        if self._relative:
            intial_investment = average_buy_price * qty * ratio_owned * self._transaction_cost
            gain = (average_sell_price * qty * (2 - self._transaction_cost) -
                     intial_investment -
                    debt)
            return gain, 100 * gain / intial_investment
        else:
            initial_investment = average_buy_price * qty * ratio_owned + number_of_entry * self._transaction_cost
            gain = ((average_sell_price * qty - self._transaction_cost) -initial_investment - debt)
            return gain, 100 * gain / initial_investment

    def _getCost(self, trade: Trade, include_borrow: bool = False, sell: bool = False) -> float:
        """
        Calculate the worth of a trade.
        :param trade: The trade to calculate the cost
        :param include_borrow: Whether to include the amount of borrowed shares in the calculation
        :param sell: Whether the trade is a sell or not.  If sell, we subtract the transaction cost from the worth.
        :return: The cost of the trade
        """
        amount = trade.amount + trade.amount_borrowed if include_borrow else trade.amount
        price = trade.security_price
        return self.estimateCost(price, amount, sell)

    def estimateCost(self, price: float, amount: int, sell: bool = False) -> float:
        """
        This method can e used by external classes such as Borker to estimate the cost of a trade.
        :param price: The price of the security
        :param amount: The amount of security in the transaction (Trade)
        :param sell: Whether the trade is a sell or not.  If sell, we subtract the transaction cost from the worth.
        :return: Worth of transaction - transaction cost
        """
        if self._relative:
            if sell:
                return price * amount * (2 - self._transaction_cost)
            else:
                return price * amount * self._transaction_cost
        else:
            if sell:
                return price * amount - self._transaction_cost
            else:
                return price * amount + self._transaction_cost

    def getShort(self) -> Dict[str, Position]:
        """
        To get what securities are sold short to later calculate interests
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._short)

    def getLong(self) -> Dict[str, Position]:
        """
        To get what securities are bought long to later calculate interests (When using margin)
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._long)

    def __getitem__(self, item: str) -> Tuple[Position, Position]:
        """
        Return the positions for the given ticker: Long, Short.  If there is no position: None.
        If there is only a long position, it will return [Equity, None]
        If there is only a short position, it will return [None, Equity]
        An both: [Equity, Equity]
        :param item:  Ticker
        :return: Long, Short
        """
        out = [None, None]
        if item in self._long:
            out[0] = deepcopy(self._long[item])
        if item in self._short:
            out[1] = deepcopy(self._short[item])

        return tuple(out)

    def get_trades(self):
        """
        To get the list of trades.  (Make a deep copy)
        :return: the list of rades
        """
        return deepcopy(self._trades)

    def get_trade_count(self, exit_only: bool = True):
        """
        To get the number of trades made.
        :param exit_only: If True, Only the trade that exit a position are counted
        :return: The number of trades
        """
        if exit_only:
            return len([trade for trade in self._trades if isinstance(trade, TradeStats)])
        else:
            return len(self._trades)

    def update_time_stock_idx(self, timestep_elapsed: int = 1):
        """
        Call this method to update the time_stock_idx for each position.  It will be used to calculate the dividends due
        to the shareholders.
        :param timestep_elapsed: The number of timestep elapsed since the last call to this method
        :return: None
        """
        for pos in self._long.values():
            pos.update_time_stock_idx(timestep_elapsed)

    def get_trade_stats(self) -> dict:
        """
        To get the stats for each trade.  (Make a deep copy)
        :return: The dictionary of trades: {
            "best_trade": float,    # In percentage
            "worst_trade": float,   # In percentage
            "win_rate": float,      # In percentage
            "avg_trade": float,     # In percentage
            "max_trade_duration": float,    # In days
            "min_trade_duration": float,    # In days
            "avg_trade_duration": float,    # In days
            "profit_factor": float, # Total gains / Total losses
            "SQN": float,           # System Quality Number
        }
        """
        rel_profit = [trade.rel_profit for trade in self._trades if isinstance(trade, TradeStats)]
        abs_profit = np.array([trade.profit for trade in self._trades if isinstance(trade, TradeStats)],
                              dtype=np.float32)
        duration_seconds = [trade.duration.total_seconds() for trade in self._trades if isinstance(trade, TradeStats)]
        # df = pd.DataFrame({"Relative Profit": rel_profit, "Absolute Profit": abs_profit, "Duration": duration})
        best_trade = max(rel_profit) if len(rel_profit) > 0 else None
        worst_trade = min(rel_profit) if len(rel_profit) > 0 else None
        win_rate = 100 * len([x for x in rel_profit if x > 0]) / len(rel_profit) if len(rel_profit) > 0 else None
        avg_trade = sum(rel_profit) / len(rel_profit) if len(rel_profit) > 0 else None
        if len(duration_seconds) == 0:
            max_trade_duration = None
            min_trade_duration = None
            avg_trade_duration = None
        else:
            max_trade_duration = max(duration_seconds) / 86_400  # In days
            min_trade_duration = min(duration_seconds) / 86_400  # In days
            avg_trade_duration = (sum(duration_seconds) / len(duration_seconds)) / 86_400  # In days
        total_gains = abs_profit[abs_profit >= 0].sum()
        total_losses = abs_profit[abs_profit < 0].sum()
        rel_profit = np.array(rel_profit, dtype=np.float32)
        if len(rel_profit) < 2:
            sqn = None
        else:
            sqn = np.sqrt(self.get_trade_count(exit_only=True)) * (rel_profit.mean() / 100) / (
                        (rel_profit / 100).std())
        if total_losses == 0:
            profit_factor = total_gains
        else:
            profit_factor = total_gains / abs(total_losses)
        return {
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "max_trade_duration": max_trade_duration,
            "min_trade_duration": min_trade_duration,
            "avg_trade_duration": avg_trade_duration,
            "profit_factor": profit_factor,
            "SQN": sqn
        }

    def get_state(self) -> dict:
        """
        Return the state of the portfolio.  (Make a deep copy)
        :return: The state of the portfolio as a dict
        """
        # Debt record is saved by the Broker
        return {
            "type": "Portfolio",
            "long": {ticker: pos.export() for ticker, pos in self._long.items()},
            "short": {ticker: pos.export() for ticker, pos in self._short.items()},
            "trades": [{"type": trade.__class__.__name__, **trade.export()} for trade in self._trades],
            "transaction_cost": self._transaction_cost,
            "transaction_relative": self._relative,
            "transaction_ids": list(self._transaction_ids),
        }

    @classmethod
    def load_state(cls, data: dict, debt_record: Dict[str, float]):
        """
        Load the state of the portfolio from a dictionary
        :param data: The dictionary to load from
        :param debt_record: The amount of debt used to buy securities: {security: amount}.  Passed by reference from broker.
        :return: The portfolio object
        """
        # Debt record is loaded by the broker
        self = cls(data["transaction_cost"], data["transaction_relative"], debt_record)
        self._long = {ticker: Position.load(pos) for ticker, pos in data["long"].items()}
        self._short = {ticker: Position.load(pos) for ticker, pos in data["short"].items()}
        self._trades = [TradeStats.load(trade) if trade["type"] == "TradeStats" else Trade.load(trade) for trade in data["trades"]]
        self._transaction_ids = set(data["transaction_ids"])
        self._long_len = len([pos for pos in self._long.values() if pos.amount > 0])
        self._short_len = len([pos for pos in self._short.values() if pos.amount > 0])
        return self

    def empty(self) -> bool:
        """
        To check if the portfolio is empty
        :return: True if the portfolio is empty, False otherwise
        """
        return len(self.long) == 0 and len(self.short) == 0

    @property
    def long(self):
        """
        Return the long positions that are not empty (amount > 0)
        """
        return {ticker: pos for ticker, pos in self._long.items() if pos.amount > 0}

    @property
    def short(self):
        """
        Return the short positions that are not empty (amount > 0)
        """
        return {ticker: pos for ticker, pos in self._short.items() if pos.amount > 0}

    def __eq__(self, other):
        return (self._long == other._long and
                self._short == other._short and
                self._trades == other._trades and
                self._transaction_cost == other._transaction_cost and
                self._relative == other._relative,
                self._debt_record == other._debt_record,
                self._transaction_ids == other._transaction_ids)

import numpy as np

from .trade import Trade, BuyLong, BuyShort, SellLong, SellShort, TradeType
from copy import deepcopy
from typing import Dict, Tuple, List, Union
from datetime import datetime, timedelta

class Position:
    """
    Data class holding info about a position
    """
    def __init__(self, ticker: str, amount: int, long: bool, average_price: float, average_filled_time: datetime, margin: bool = False):
        self.ticker = ticker
        self.amount = amount
        self.average_price = average_price
        self.on_margin = margin
        self.long = long
        if amount <= 0:
            raise ValueError("Amount cannot be negative nor zero")
        if not long and not margin:
            raise ValueError("Short position must be on margin")
        self.average_filled_time = average_filled_time
        self.last_dividends_dt = average_filled_time

        # This index correspond to the amount of shares hold times the time hold.  It will be used to calculate the
        # dividends due to the shareholders.
        # Formula: days * (amount)
        self.time_stock_idx = 0

    @property
    def purchase_worth(self):
        return self.average_price * self.amount

    def __str__(self):
        mrg = "MARGIN" if self.on_margin else ""
        lg = "LONG" if self.long else "SHORT"
        return f"EQUITY: {self.amount}x{self.ticker} {round(self.average_price, 2)}$ {mrg} {lg}"

    def __repr__(self):
        return f"EQUITY: {self.ticker}"

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
            return new

        elif isinstance(other, Trade):
            if self.ticker != other.security:
                raise ValueError("Cannot add positions of different tickers")
            if self.long:
                if other.trade_type == TradeType.SellLong:
                    raise ValueError("Cannot add a trade that is selling to long position")
                elif other.trade_type != TradeType.BuyLong:
                    raise ValueError("Invalid trade type for addition to long position")
            else:    # Short
                if other.trade_type == TradeType.BuyShort:
                    raise ValueError("Cannot add a trade that is buying to short position")
                elif other.trade_type != TradeType.SellShort:
                    raise ValueError("Invalid trade type for addition to short position")

            new = deepcopy(self)
            # Newly acquired position.  (Was not held before)
            current_amount = self.amount
            other_amount = other.amount + other.amount_borrowed
            total_amount = current_amount + other_amount
            new.average_price = ((current_amount / total_amount) * self.average_price +
                                  (other_amount / total_amount) * other.security_price)
            new.amount += other.amount + other.amount_borrowed
            # time stock idx is not updated when adding a trade because the position was not held before.
            new.time_stock_idx += 0
            new.average_filled_time = (self.average_filled_time +
                                            timedelta(seconds=(other.timestamp -
                                                               self.average_filled_time).total_seconds() / 2))

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
        borrowed.
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
            else:    # Short
                if other.trade_type == TradeType.BuyShort:
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
            "amount": self.amount,
            "long": self.long,
            "average_price": self.average_price,
            "on_margin": self.on_margin,
            "average_filled_time": str(self.average_filled_time),
            "last_dividends_dt": str(self.last_dividends_dt),
            "time_stock_idx": self.time_stock_idx
        }
    @classmethod
    def load(cls, data: dict):
        """
        Load the object from a dictionary
        :param data: The dictionary to load from
        :return: The object
        """
        self = cls(data["ticker"], data["amount"], data["long"], data["average_price"],
                   datetime.fromisoformat(data["average_filled_time"]), margin=data["on_margin"])
        self.last_dividends_dt = datetime.fromisoformat(data["last_dividends_dt"])
        self.time_stock_idx = data["time_stock_idx"]
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
    def __init__(self, trade: Trade, duration: timedelta, profit: float, rel_profit: float):
        """
        :param trade: The trade object (SellLong or BuyShort)
        :param duration: The duration of the trade
        :param profit: The profit made on the trade
        :param rel_profit: The profit made on the trade relative to the amount invested NOT IN PERCENTAGE
        :raise ValueError: If the trade type is not SellLong or BuyShort (Doesn't exit a position)
        """
        if trade.trade_type != TradeType.SellLong and trade.trade_type != TradeType.BuyShort:
            raise ValueError(f"Invalid trade type for TradeStats.  The tradetype must exit a position.  "
                             f"Got {trade.trade_type}, but expected SellLong or BuyShort.")
        self.trade = trade
        self.duration = duration
        self.profit = profit
        self.rel_profit = rel_profit

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
            "rel_profit": self.rel_profit
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
        return cls(trade, duration, data["profit"], data["rel_profit"])

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
    def __init__(self, transaction_cost: float = 10., transaction_relative: bool = False, debt_record: Dict[str, float] = {}):
        """
        :param transaction_cost: The cost of a transaction (buy or sell) in $ or in %
        :param transaction_relative: Whether the transaction cost is in percentage relative to transaction cost or fix price
        :param debt_record: The amount of debt used to buy securities: {security: amount}.  Passed by reference from broker.
        """
        # Keys will be security (ticker) and the value will be Equity data object
        self._long: Dict[str, Position] = {}
        self._short: Dict[str, Position] = {}
        self._trades: List[Union[TradeStats, Trade]] = []    # TradeStats when closing trade and Trade when opening one.
        self._transaction_cost = transaction_cost
        self._relative = transaction_relative
        self._debt_record: Dict[str, float] = debt_record
        self._transaction_ids = set()

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

        if isinstance(trade, BuyLong):
            if trade.security in self._long:
                self._long[trade.security] += trade
            else:
                self._long[trade.security] = Position(trade.security, trade.amount + trade.amount_borrowed, True,
                                                trade.security_price, trade.timestamp, margin=trade.amount_borrowed > 0)

            # Handle debt record
            if trade.security in self._debt_record:
                if self._relative:
                    self._debt_record[trade.security] += trade.amount_borrowed * trade.security_price * self._transaction_cost
                else:
                    self._debt_record[trade.security] += trade.amount_borrowed * trade.security_price + self._transaction_cost
            else:
                if self._relative:
                    self._debt_record[trade.security] = trade.amount_borrowed * trade.security_price * self._transaction_cost
                else:
                    self._debt_record[trade.security] = trade.amount_borrowed * trade.security_price + self._transaction_cost
            # Save for stats
            self._trades.append(trade)
            return -self.getCost(trade)
        elif isinstance(trade, SellLong):
            if trade.security not in self._long:
                raise RuntimeError("Cannot sell Long if the security is not acquired.")
            elif self._long[trade.security].amount< trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot sell Long more securities than the portfolio has")
            else:
                # Get relative debt (Debt that will be repaid to the bank).
                n_assets = trade.amount + trade.amount_borrowed
                total_assets = self._long[trade.security].amount
                relative_debt = self._debt_record[trade.security] * (n_assets / total_assets)
                duration = trade.timestamp - self._long[trade.security].average_filled_time
                average_buy_price = self._long[trade.security].average_price
                profit, rel_profit = self.getLongProfit(average_buy_price,
                                                   trade.security_price,
                                                   trade.amount + trade.amount_borrowed,
                                                   relative_debt)
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               profit, rel_profit))
                # Handle debt record
                self._debt_record[trade.security] -= relative_debt    # Debt that has been repaid
                # Should not be negative (Or slighly because of float precision)
                if self._debt_record[trade.security] <= 0:
                    del self._debt_record[trade.security]
                self._long[trade.security] -= trade
                if self._long[trade.security].amount == 0:
                    del self._long[trade.security]

                return self.getCost(trade)

        elif isinstance(trade, SellShort):
            if trade.security in self._short:
                self._short[trade.security] += trade.amount_borrowed
            else:
                self._short[trade.security] = Position(trade.security, trade.amount_borrowed, False,
                                                     trade.security_price, trade.timestamp, margin=True)
            self._trades.append(trade)
        elif isinstance(trade, BuyShort):
            if trade.security not in self._short:
                raise RuntimeError("Cannot buy short if the security has not been sold short.")
            elif self._short[trade.security].amount < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot buy short more securities than the portfolio has sold short.")
            else:
                duration = trade.timestamp - self._long[trade.security].average_filled_time
                average_sell_price = self._short[trade.security].average_price
                trade.amount = 0    # Just to make sure
                self._short[trade.security] -= trade
                if self._short[trade.security].amount == 0:
                    del self._short[trade.security]
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               *self.getShortProfit(average_sell_price,trade.security_price, trade.amount_borrowed)))

    def getShortProfit(self, average_sell_price: float, average_buy_price: float, qty: int) -> Tuple[float, float]:
        """
        Calculate the profit made on a short trade
        :param average_sell_price: The price at which the security was sold
        :param average_buy_price: The price at which the security was bought
        :param qty: The number of shares in the trade
        :return: The profit (positive) or loss (negative) made on the trade, relative profit
        """
        if self._relative:
            gain = (average_buy_price*qty - average_sell_price*qty) * (1 - self._transaction_cost)
            return gain, 100 * gain / average_sell_price
        else:
            gain = (average_buy_price*qty - average_sell_price*qty) - self._transaction_cost
            return gain, 100 * gain / average_sell_price

    def getLongProfit(self, average_buy_price: float, average_sell_price: float, qty: int, debt) -> Tuple[float, float]:
        """
        Calculate the profit made on a long trade
        :param average_buy_price: The price at which the security was bought
        :param average_sell_price: The price at which the security was sold
        :param qty: The number of shares in the trade
        :param debt: The amount of debt used to buy the security
        :return: The profit (positive) or loss (negative) made on the trade, relative profit in percentage
        """
        if self._relative:
            gain = (average_sell_price*qty - average_buy_price*qty) * (1 - self._transaction_cost) - debt
            return gain, 100 * gain / average_buy_price
        else:
            gain = (average_sell_price*qty - average_buy_price*qty) - self._transaction_cost - debt
            return gain, 100 * gain / average_buy_price

    def getCost(self, trade: Trade) -> float:
        """
        Calculate the cost of a trade
        :param trade: The trade to calculate the cost
        :return: The cost of the trade
        """
        if self._relative:
            return trade.security_price * trade.amount * self._transaction_cost
        else:
            return trade.security_price * trade.amount + self._transaction_cost

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
        abs_profit = np.array([trade.profit for trade in self._trades if isinstance(trade, TradeStats)], dtype=np.float32)
        duration_seconds = [trade.duration.total_seconds() for trade in self._trades if isinstance(trade, TradeStats)]
        # df = pd.DataFrame({"Relative Profit": rel_profit, "Absolute Profit": abs_profit, "Duration": duration})
        best_trade = max(rel_profit)
        worst_trade = min(rel_profit)
        win_rate = len([x for x in rel_profit if x > 0]) / len(rel_profit)
        avg_trade = sum(rel_profit) / len(rel_profit)
        max_trade_duration = max(duration_seconds) / 86_400    # In days
        min_trade_duration = min(duration_seconds) / 86_400    # In days
        avg_trade_duration = (sum(duration_seconds) / len(duration_seconds)) / 86_400    # In days
        total_gains = abs_profit[abs_profit > 0].sum()
        total_losses = abs_profit[abs_profit < 0].sum()
        rel_profit = np.array(rel_profit, dtype=np.float32)
        sqn = np.sqrt(self.get_trade_count(exit_only=True)) * (rel_profit.mean() / 100) / ((rel_profit / 100).std() or np.nan)
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
            "transaction_relative": self._relative
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
        self._trades = [Trade.load(trade) for trade in data["trades"]]
        return self

    def empty(self) -> bool:
        """
        To check if the portfolio is empty
        :return: True if the portfolio is empty, False otherwise
        """
        return len(self._long) == 0 and len(self._short) == 0


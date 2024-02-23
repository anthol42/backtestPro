import numpy as np
from .account import Account
from .trade import BuyLongOrder, SellLongOrder, BuyShortOrder, SellShortOrder, TradeOrder, TradeType
from .portfolio import Portfolio, Position
from datetime import datetime, date
from typing import Dict, Tuple, List, Optional, Set
from copy import deepcopy
import numpy.typing as npt
from .tsData import DividendFrequency


class MarginCall:
    """
    Data class to store info about margin calls
    """
    def __init__(self, time_remaining: int, amount: float):
        """

        :param time_remaining: Time remaining in number of steps until liquidation
        :param amount: Amount in $ that needs to be added to account to come out of margin call
        """
        self.time_remaining = time_remaining
        self.amount = amount

    def __str__(self):
        return f"MARGIN CALL {self.time_remaining} steps for {round(self.amount, 2)}$"

    def __repr__(self):
        return f"MARGINCALL({self.time_remaining}, {self.amount})"

    def export(self) -> dict:
        """
        This method export the margin call object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": "MarginCall",
            "time_remaining": self.time_remaining,
            "amount": self.amount
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method load a margin call object from a dictionary.
        :param data: The dictionary containing the object state
        :return: The object state as a dictionary
        """
        return MarginCall(data["time_remaining"], data["amount"])

    def __eq__(self, other):
        if isinstance(other, int):
            return self.time_remaining == other
        elif isinstance(other, MarginCall):
            return self.time_remaining == other.time_remaining and self.amount == other.amount
        else:
            NotImplementedError(f"Equal operator not implemented for type: {type(other)}")


class BrokerState:
    """
    Data class to store message from broker to strategy
    """
    def __init__(self, margin_calls: Dict[str, MarginCall], bankruptcy: bool):
        self.margin_calls = margin_calls
        self.bankruptcy = bankruptcy

    def export(self):
        """
        This method export the broker state object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": "BrokerState",
            "margin_calls": {key: value.export()
                             for key, value in self.margin_calls.items()},
            "bankruptcy": self.bankruptcy
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method load a broker state object from a dictionary.
        :param data: The dictionary containing the object state
        :return: The object state as a dictionary
        """
        return BrokerState({key: MarginCall.load(value) for key, value in data["margin_calls"].items()}, data["bankruptcy"])

    def __eq__(self, other):
        return self.margin_calls == other.margin_calls and self.bankruptcy == other.bankruptcy

class StepState:
    """
    Record the state of the broker at each steps for easier strategy debugging
    """
    def __init__(self, timestamp: datetime, worth: float, pending_orders: List[TradeOrder], filled_orders: List[TradeOrder],
                 margin_calls: Dict[str, MarginCall]):
        """
        :param timestamp: The date and time of the current step
        :param worth: The worth of the portfolio
        :param pending_orders: The pending orders that were not executed
        :param filled_orders: The orders that were executed at the current step
        :param margin_calls: The margin calls active at the current step
        """
        self.timestamp = timestamp
        self.worth = worth
        self.pending_orders = pending_orders
        self.filled_orders = filled_orders
        self.margin_calls = margin_calls

    def export(self) -> dict:
        """
        This method export the step state object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": "StepState",
            "timestamp": str(self.timestamp),
            "worth": self.worth,
            "pending_orders": [order.export() for order in self.pending_orders],
            "filled_orders": [order.export() for order in self.filled_orders],
            "margin_calls": {key: value.export()
                             for key, value in self.margin_calls.items()}
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method load a step state object from a dictionary.
        :param data: The dictionary containing the object state
        :return: The object state as a dictionary
        """
        return StepState(datetime.fromisoformat(data["timestamp"]), data["worth"],
                         [TradeOrder.load(order) for order in data["pending_orders"]],
                         [TradeOrder.load(order) for order in data["filled_orders"]],
                         {key: MarginCall.load(value) for key, value in data["margin_calls"].items()})

    def __eq__(self, other):
        return (self.timestamp == other.timestamp and
                self.worth == other.worth and
                self.pending_orders == other.pending_orders and
                self.filled_orders == other.filled_orders and
                self.margin_calls == other.margin_calls)


class Broker:
    """
    When using margin, you need to add a column in the data called 'marginable' that is bool corresponding if a stock
    can have margin or not.
    Note: If there is not enough cash in the account when monthly fees are deduced, the strategy fills a bankruptcy, and
    the simulation stops.
    """
    def __init__(self, bank_account: Account, commission: float = None,
                 relative_commission: float = None, margin_interest: float = 0,
                 min_initial_margin: float = 0.5, min_maintenance_margin: float = 0.25,
                 liquidation_delay: int = 2, min_initial_margin_short: float = 0.5,
                 min_maintenance_margin_short: float = 0.25):
        """

        :param bank_account: The bank account holding cash and collateral with the initial cash set.
        :param commission: The commission to be paid for each trade.  This parameter cannot be set if relative_commission
                            is set.  And vice versa.  If commission and relative_commission are both None, the commission
                            is set to 0.
        :param relative_commission: The commission to be paid for each trade as a ratio (0-1) of the trade value.  This
                                    parameter cannot be set if commission is set.  And vice versa.
        :param margin_interest: The margin interest rate.  It is the interest rate charged on borrowed money.  It is
                                calculated daily and charged monthly.
        :param min_initial_margin: The initial margin required to buy a stock on margin.  It is a ratio (0-1) of the
                                    stock's value.  Example: 50% initial margin is 0.5
        :param min_maintenance_margin: The minimum margin required to keep a stock on margin before a margin call.
                                        It is a ratio (0-1) of the stock's value.  Example: 25% maintenance margin is 0.25
        :param liquidation_delay: The time a margin call has to be active before the broker liquidate positions to pay
                                    the margin call. (Must be grater than 0, and an int)
        :param min_initial_margin_short: The minimum initial margin required to sell a stock short.  It is a ratio (0-1)
        :param min_maintenance_margin_short: The minimum maintenance margin required to keep a stock short before a margin
                                            call.  It is a ratio (0-1) of the stock's value.
        :raise ValueError: If both commission and relative_commission are set.
        :raise ValueError: If relative_commission is not None and not between 0 and 1
        :raise ValueError: If commission is not None and smaller than 0
        :raise ValueError: If margin_interest is not between 0 and 1
        :raise ValueError: min_initial_margin and min_maintenance_margin are not between 0 and 1
        :raise ValueError: If min_maintenance_margin_short and min_initial_margin_short are not between 0 and 1
        :raise ValueError: If min_maintenance_margin is greater than or equal to min_initial_margin
        :raise ValueError: If min_maintenance_margin_short is greater than or equal to min_initial_margin_short
        """
        if commission is not None and relative_commission is not None:
            raise ValueError("Must choose between relative commission or absolute commission!")
        if commission is None and relative_commission is None:
            commission = 0

        if relative_commission is not None:
            if relative_commission < 0. or relative_commission > 1.:
                raise ValueError(f"Relative commission must be between 0 and 1.  Got: {relative_commission}")
            self._comm = 1 + relative_commission
            self._relative = True
        else:
            if commission < 0.:
                raise ValueError(f"Commission must be greater or equal to 0.  Got: {commission}")
            self._comm = commission
            self._relative = False
        if margin_interest < 0. or margin_interest > 1.:
            raise ValueError(f"Margin interest must be between 0 and 1.  Got: {margin_interest}")
        if min_maintenance_margin > 1. or min_maintenance_margin < 0.:
            raise ValueError(f"Minimum maintenance margin must be between 0 and 1.  Got: {min_maintenance_margin}")

        if min_initial_margin > 1. or min_initial_margin < 0.:
            raise ValueError(f"Minimum initial margin must be between 0 and 1.  Got: {min_initial_margin}")

        if min_maintenance_margin >= min_initial_margin:
            raise ValueError(f"Minimum maintenance margin must be smaller or equal than minimum initial margin. ")

        if min_initial_margin_short > 1. or min_initial_margin_short < 0.:
            raise ValueError(f"Minimum initial margin for short must be between 0 and 1.  Got: {min_initial_margin_short}")

        if min_maintenance_margin_short > 1. or min_maintenance_margin_short < 0.:
            raise ValueError(f"Minimum maintenance margin for short must be between 0 and 1.  Got: {min_maintenance_margin_short}")

        if min_maintenance_margin_short >= min_initial_margin_short:
            raise ValueError(f"Minimum maintenance margin for short must be smaller or equal than minimum initial margin for short. ")


        self.min_maintenance_margin = min_maintenance_margin
        self.min_initial_margin = min_initial_margin
        self.margin_interest = margin_interest
        self.min_initial_margin_short = min_initial_margin_short
        self.min_maintenance_margin_short = min_maintenance_margin_short
        if liquidation_delay <= 0:
            raise ValueError(f"Liquidation delay must be greater than 0.  Got: {liquidation_delay}")
        self.liquidation_delay = int(liquidation_delay)

        # By-stock record of borrowed money {ticker, borrowed_amount}
        self._debt_record: Dict[str, float] = {}
        self._queued_trade_offers = []
        # Convert commission to 0-1 for Porfolio
        if self._relative:
            comm = self._comm - 1
        else:
            comm = self._comm
        self.portfolio = Portfolio(comm, self._relative, self._debt_record)  # Equities bought with available cash
        self.account = bank_account
        self.n = 0    # Keeps the count of trades.  (To give trades an id)
        self._month_interests = 0    # The current interests of the month.  They will be charged at the end of the month
        self._current_month: Optional[int] = None       # Remember the current month so the broker knows when we change
                                                        # month to charge monthly fees such as interest rates.
        self._last_day: Optional[date] = None           # Remember the last day so the broker knows how long
                                                        # elapsed between the last step and the current step to charge
                                                        # the correct amount of interest
        self._last_step: Optional[datetime] = None     # Remember the last step.

        self.message = BrokerState({}, False)

        # Stats
        self.historical_states: List[StepState] = []

        # Current timestamp (The timestamp of last datapoint sent to the strategy for the current timestep.
        # Set by BackTest object)
        self._current_timestamp: Optional[datetime] = None

        # Exposure time in days
        self.exposure_time = 0

        # This attribute stores the contribution of each position to the collateral to update the collateral faster
        # after closing a position
        self._cache = {
            "long_collateral_contribution": {},
            "short_collateral_contribution": {},
        }

    def set_current_timestamp(self, timestamp: datetime):
        self._current_timestamp = timestamp

    def buy_long(self, ticker: str, amount: int, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        self._queued_trade_offers.append(BuyLongOrder(self._current_timestamp, ticker, price_limit, amount,
                                                      amount_borrowed, expiry))

    def sell_long(self, ticker: str, amount: int, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        self._queued_trade_offers.append(SellLongOrder(self._current_timestamp, ticker, price_limit, amount, amount_borrowed, expiry))

    def sell_short(self, ticker: str, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        self._queued_trade_offers.append(SellShortOrder(self._current_timestamp, ticker, price_limit, 0, amount_borrowed, expiry))

    def buy_short(self, ticker: str, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        self._queued_trade_offers.append(BuyShortOrder(self._current_timestamp, ticker, price_limit, 0, amount_borrowed, expiry))

    def tick(self, timestamp: datetime, next_timestamp, security_names: List[str], current_tick_data: np.ndarray,
             next_tick_data: np.ndarray, marginables: npt.NDArray[bool], dividends: npt.NDArray[np.float32],
             div_freq: List[DividendFrequency], short_rates: npt.NDArray[np.float32]):
        """
        The simulation calls this method after the strategy has run.  It will calculate interests and margin call if
        applicable.  It will do trades that can be done in the trade queue at the next open.
        :param timestamp: The date and time of the current step
        :param next_timestamp: The date and time of the next step where orders will be evaluated/bought
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array of prices of each security for the current step(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :param marginables: A boolean array of shape (n_securities, 2) [Marginable, Shortable) where True means that
                            the security can be bought on margin / sold short and False means that it cannot be bought on
                            margin / sold short.
        :param dividends: A float array of shape (n_securities, ) containing the dividends of each security for the
                            current step.
        :param div_freq: The frequency that the security is paying dividends.
        :param short_rates: The interest rates for each security that the user held overnight.  Shape(n_securities, )
        :return: None
        """

        # If it is the first step, we need to initialize the portfolio with the current month
        if self._current_month is None:
            self._current_month = timestamp.month

        if self._last_day is None:
            self._last_day = timestamp.date()
        else:
            n_days_elapsed_since_last_step = (timestamp.date() - self._last_day).days
            # We update the time stock index for each stock in the portfolio to know how long each asset has been held
            # for later dividend calculation
            self.portfolio.update_time_stock_idx(n_days_elapsed_since_last_step)

        if self._last_step is None:
            self._last_step = timestamp
        else:
            # If portfolio is not empty, we add the time elapsed since the last step to the exposure time
            if not self.portfolio.empty():
                self.exposure_time += (timestamp - self._last_step).total_seconds() / 86_400


        # Step 1: Get the total borrowed money, cash and portfolio worth
        borrowed_money = self._get_borrowed_money()

        # Cash in dividends
        self._cashin_dividends(timestamp, security_names, dividends, div_freq)

        # Evaluate the worth of the portfolio
        worth = self._get_worth(security_names, current_tick_data)

        # Step 2: If the portfolio has borrowed money: we calculate current interests and add them to monthly interests.
        # Interest rates are calculated daily but charged monthly.
        self._update_interests(timestamp, borrowed_money)
        self._update_interests_short(timestamp, next_timestamp, security_names, current_tick_data, short_rates)

        # Step 3: Update the account collateral after paying debts and interests, if it wasn't paid the previous steps
        self._pay_missing_funds(timestamp)
        self._update_account_collateral(timestamp, security_names, current_tick_data)


        # Step 4: If there is borrowed money, reduce the delay of margin calls.  (It is normal that newly initialized
        # margin call's delay will already be reduced.  It is took into account in the liquidation process)
        self._decrement_margin_call()

        # Step 5: Liquidate expired margin calls
        self._liquidate_expired_mc(timestamp, security_names, next_tick_data)

        # Step 6: Execute trades that can be executed
        filled_orders = self._execute_trades(next_timestamp, security_names, next_tick_data, marginables)

        # Step 7: Charge interests if it's the first day of the month
        # Interest are deducted from account.  If there is not enough money in the account to payout interests,
        # the account, we create a new margin call for missing funds and interests will be charged on these because.
        self._charge_interests(timestamp)

        # Step 8: Save states
        self.historical_states.append(
            StepState(timestamp, worth, self._queued_trade_offers, filled_orders, self.message.margin_calls)
        )

        # Update states
        self._last_day = timestamp.date()
        self._last_step = timestamp

    def _liquidate_expired_mc(self, timestamp: datetime, security_names: List[str], next_tick_data: np.ndarray):
        """
        Liquidate expired margin calls.  Long and short
        :param timestamp: The date and time of the current step
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
        :return: None
        """
        # Long evaluation and liquidation
        pos_to_liquidate = [ticker for ticker in self.message.margin_calls if self.message.margin_calls[ticker] == -1]
        for call_name in pos_to_liquidate:
            if call_name.startswith("missing_funds") or call_name == "short margin call":
                continue
            ticker = call_name.replace("long margin call ", "")
            # We sell at this price if it is long
            long, _ = self.portfolio[ticker]  # Can be long AND short even though it does not make that much sense
            if long is not None:
                self.sell_long(ticker, long.amount, 0, None, (None, None))

        # Short evaluation and liquidation
        if self.message.margin_calls.get("short margin call") and self.message.margin_calls["short margin call"].time_remaining == -1:
            self._liquidate(self.message.margin_calls["short margin call"].amount, timestamp, security_names,
                            next_tick_data)

    def _charge_interests(self, timestamp: datetime):
        """
        Charge interests to the account at each new month.  Plus it updates the current month
        :param timestamp: The date and time of the current step
        :return: None
        """
        if timestamp.month != self._current_month:
            self._current_month = timestamp.month
            if self.account.get_cash() > self._month_interests:
                self.account.withdrawal(self._month_interests, timestamp, comment="Interest payment")
                self._month_interests = 0
            else:
                if self.account.get_cash() > 0:
                    # Put interests as debt and as margin call that the user needs to pay
                    self.new_margin_call(self._month_interests - self.account.get_cash())
                    self._month_interests = 0
                    self.account.withdrawal(self.account.get_cash(), timestamp,
                                            comment="Interest payment")
                else:
                    self.new_margin_call(self._month_interests)
                    self._month_interests = 0

    def _decrement_margin_call(self):
        """
        Decrement the margin call delay until liquidation.  If it reaches -1, the position is
        liquidated(not in this method)
        :return:
        """
        # Decrement margin call delay until liquidation.  If it reaches -1, the position is liquidated
        for key in self.message.margin_calls:
            self.message.margin_calls[key].time_remaining -= 1

    def _pay_missing_funds(self, timestamp: datetime):
        """
        If there are missing funds, it takes the remaining cash in the account and pays the missing funds.
        :param timestamp: The date and time of the current step
        :return: None
        """
        if "missing_funds" not in self.message.margin_calls:
            return
        if self.account.get_cash() > self.message.margin_calls["missing_funds"].amount:
            self.account.withdrawal(self.message.margin_calls["missing_funds"].amount, timestamp,
                                    comment="Missing funds")
            self.remove_margin_call("missing_funds")
        else:
            if self.account.get_cash() > 0:
                self.message.margin_calls["missing_funds"].amount -= self.account.get_cash()
                self._debt_record["missing_funds"] -= self.account.get_cash()
                self.account.withdrawal(self.account.get_cash(), timestamp, comment="Missing funds")

    def _get_borrowed_money(self) -> float:
        """
        Get the total borrowed money.  (shares bought on margin, and missing funds)
        :return: The total borrowed money
        """
        return sum([amount for name, amount in self._debt_record.items() if "margin call" not in name])

    def _update_interests_short(self, timestamp: datetime, next_timestamp: datetime, security_names: List[str],
                                current_tick_data: np.ndarray, short_rates: npt.NDArray[np.float32]):
        """
        The counterpart of _update_interests for short positions.  It will update the interests of the account based on
        the securities the user held overnight.  The interests rates are provided from a timeseries by the backtest,
        object.  (Passed through the tick method of the broker, then here.)
        Note:
            This method expects to be called before executing trade orders.
        :param timestamp: The current timestamp.
        :param next_timestamp: The next timestamp.  (Where orders will be evaluated/completed)
        :param security_names: Aa list of all securities where the index of each ticker is the index of the data of the
                                 corresponding security in the 'current_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
        :param short_rates: The interest rates for each security that the user held overnight.  Shape(n_securities, )
        :return: None
        """
        if timestamp.date() == next_timestamp.date():    # Intraday
            return

        delta = (next_timestamp.date() - timestamp.date()).days
        for ticker in self.portfolio.getShort():
            if self.portfolio.getShort()[ticker].amount > 0:
                ticker_idx = security_names.index(ticker)
                rate = short_rates[ticker_idx]
                mk_value = current_tick_data[ticker_idx, 3] * self.portfolio.getShort()[ticker].amount
                self._month_interests += delta * rate * mk_value / 365

    def _update_interests(self, timestamp: datetime, borrowed_money):
        """
        Update the interests of the account.  If the amount of borrowed money is 0, it won't do nothing.
        :param timestamp: The date and time of the current step
        :param borrowed_money: The amount of money borrowed.  (Shares sold short, bought on margin, and missing funds)
        :return: None
        """
        days_elapsed: int = (timestamp.date() - self._last_day).days
        if borrowed_money > 0:
            self._month_interests += days_elapsed * self.margin_interest * borrowed_money / 365
    def _cashin_dividends(self, timestamp: datetime, security_names: List[str], dividends: npt.NDArray[np.float32],
                            div_freq: List[DividendFrequency]):
            """
            Cash in dividends
            :param timestamp: The date and time of the current step
            :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                                 corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
            :param dividends: A float array of shape (n_securities, ) containing the dividends of each security for the
                                current step.
            :param div_freq: The frequency that the security is paying dividends.
            :return: None
            """
            for ticker, position in self.portfolio.getLong().items():
                ticker_idx = security_names.index(ticker)
                if dividends[ticker_idx] > 0:
                    dividend_payout = self.compute_dividend_payout(position, div_freq[ticker_idx],
                                                                   dividends[ticker_idx])
                    position.dividends_got_paid(timestamp)
                    self.account.deposit(dividend_payout,
                                         timestamp, comment=f"Dividends - {ticker}")

    def _execute_trades(self, next_timestep: datetime, security_names: List[str], next_tick_data: np.ndarray,
                        marginables: npt.NDArray[bool]) -> List[TradeOrder]:
        """
        Execute trades that can be executed.  (Price becomes within limits)
        :param next_timestep: The next timestep timestamp (Where orders will be evaluated/completed)
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :param marginables: A boolean array of shape (n_securities, 2) [Marginable, Shortable) where True means that
                            the security can be bought on margin / sold short and False means that it cannot be bought on
                            margin / sold short.
        :return: Teh filled orders
        """
        filled_orders = []
        expired_orders = []
        for order in self._queued_trade_offers:
            if order.expiry is not None and order.expiry < next_timestep:
                expired_orders.append(order)
                continue

            eq_idx = security_names.index(order.security)

            result = self.make_trade(order, next_tick_data[eq_idx], next_timestep, marginables[eq_idx, 0], marginables[eq_idx, 1])
            if result:
                filled_orders.append(order)

        # At the end of the timestep, when all trades that could have been done are executed, we update the collateral.
        self._update_account_collateral(next_timestep, security_names, next_tick_data, message="Trade execution")

        if self.account.get_cash() > 0 and "missing_funds" in self.message.margin_calls:
            if self.message.margin_calls["missing_funds"].amount < self.account.get_cash():
                self.account.withdrawal(self.message.margin_calls["missing_funds"].amount, next_timestep,
                                        comment="Paying Missing funds after doing trades")
                self.remove_margin_call("missing_funds")
            else:
                self.message.margin_calls["missing_funds"].amount -= self.account.get_cash()
                self._debt_record["missing_funds"] -= self.account.get_cash()
                self.account.withdrawal(self.account.get_cash(), next_timestep, comment="Paying Missing funds after doing trades")

        # Remove filled orders from pending orders
        for order in filled_orders:
            self._queued_trade_offers.remove(order)

        # Remove expired orders from pending orders
        for order in expired_orders:
            self._queued_trade_offers.remove(order)

        return filled_orders


    @staticmethod
    def compute_dividend_payout(position: Position,
                                div_freq: DividendFrequency, dividends: float) -> float:
        """
        Compute the dividend payout of a long position
        :param position: The position to calculate dividend payout
        :param div_freq: The frequency that the security is paying dividends.
        :param dividends: The actual dividends payout for 1 share and for the period in div_freq.
        :return: Dividend payout
        :raise RuntimeError: If the position is not long
        """
        if not position.long:
            raise RuntimeError("Dividend payout can only be calculated for long positions.")
        hold_idx = position.time_stock_idx    # Equivalent to time * num_share_held

        # If I divide the hold_idx by the time, I get the average number of shares held for the period.

        if div_freq == DividendFrequency.MONTHLY:
            avg_amount = hold_idx / 30
        elif div_freq == DividendFrequency.QUARTERLY:
            avg_amount = hold_idx / 90
        elif div_freq == DividendFrequency.BIANNUALLY:
            avg_amount = hold_idx / 180
        else:  # DividendFrequency.YEARLY
            avg_amount = hold_idx / 365

        return dividends * avg_amount

    def _get_short_collateral(self, total_cash: float, security_names: List[str],
                              current_tick_data: npt.NDArray[float]) -> float:
        """
        Get the total collateral of short positions  It will also update margin calls at the same time.
        If the account doesn't have enough money, a margin call must be emitted.
        If a short margin call already exists, its value will be updated.  Otherwise, a new one will be created.
        :param total_cash: The total cash in the account.  (Do not remove collateral)
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the current step shape(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: The total collateral of short positions
        """
        collateral = 0.
        for ticker in self.portfolio.getShort():
            if self.portfolio.getShort()[ticker].amount == 0:
                continue
            eq_idx = security_names.index(ticker)
            o, h, l, close = tuple(current_tick_data[eq_idx].tolist()) # (Open, High, Low, Close)

            # Calculate market value of the short position in a situation where we would want to buy back the position.
            # In that case, we need to add the commission to the market value.
            if self._relative:
                mk_value = self.portfolio.getShort()[ticker].amount * close * self._comm
            else:
                mk_value = self.portfolio.getShort()[ticker].amount * close + self._comm
            contrib = (1 + self.min_maintenance_margin_short) * mk_value
            collateral += contrib
            self._cache["short_collateral_contribution"][ticker] = contrib

        if total_cash - collateral < 0:
            if "short margin call" in self.message.margin_calls:
                self.message.margin_calls["short margin call"].amount = collateral - total_cash
                self._debt_record["short margin call"] = collateral - total_cash
            else:
                self.new_margin_call(collateral - total_cash, "short margin call")

            return total_cash
        else:
            if "short margin call" in self.message.margin_calls:
                self.remove_margin_call("short margin call")
            return collateral

    def _get_worth(self, security_names: List[str], current_tick_data: np.ndarray) -> float:
        """
        Get the worth of the portfolio.  (Doesn't include the transaction cost as if we would sell the positions)
        Formula:
            worth = cash + sum(long_positions) - sum(short_positions)

            where:
                long_positions = sum(long_position_amount * close_price - debt)
                short_positions = sum(short_position_amount * close_price)
                cash: The total cash in portfolio excluding all collateral.
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                                 corresponding security in the 'current_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the current step shape(n_securities, 4)
        :return: worth in dollars
        """
        worth = self.account.get_total_cash()
        for ticker, position in self.portfolio.getLong().items():
            eq_idx = security_names.index(ticker)
            price = tuple(current_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
            close = price[3]
            debt = self._debt_record.get(ticker)
            if debt is None:
                debt = 0
            worth += position.amount * close - debt

        for ticker, position in self.portfolio.getShort().items():
            eq_idx = security_names.index(ticker)
            price = tuple(current_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
            close = price[3]
            worth -= position.amount * close

        return worth


    def _get_long_collateral(self, available_cash: float, security_names: List[str],
                             current_tick_data: npt.NDArray[float]) -> float:
        """
        Get the total collateral of long positions (Only the one having a margin call impact the collateral requirement)
        It will also update margin calls at the same time ( remove the ones that aren't called anymore and create the
        new ones.)
        Note
            You should run this method after the short collateral has been evaluated.

        :param available_cash: The available cash in the account.  (Total - reserved for short collateral)
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the current step shape(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: the collateral of long positions
        """
        new_calls = set()
        collateral = 0.
        for ticker in list(self._debt_record.keys()):    # We only look at the positions that have borrowed money (margin trades)
            if ticker.startswith("missing_funds") or ticker == "short margin call" or ticker.startswith("long margin call"):
                continue
            long, _ = self.portfolio[ticker]

            eq_idx = security_names.index(ticker)
            if long is not None:
                if long.ticker in self.message.margin_calls:
                    continue
                if long.amount == 0:
                    continue
                # Verify if the stock is in margin call for long
                is_margin_call, amount = self._isMarginCall(long.amount * current_tick_data[eq_idx, -1],
                                                            self._debt_record[ticker],
                                                            self.min_maintenance_margin,
                                                            self._comm, self._relative)
                if is_margin_call:
                    # Check if true margin call compared with account balance
                    if amount > available_cash:
                        # Only adds it if not in margin call already.  (to not reset the delay)
                        if f"long margin call {ticker}" not in self.message.margin_calls:
                            if available_cash > 0:
                                self.new_margin_call(amount - available_cash, f"long margin call {ticker}")
                                collateral += available_cash
                                self._cache["long_collateral_contribution"][ticker] = available_cash
                                available_cash = 0
                            else:
                                self.new_margin_call(amount, f"long margin call {ticker}")
                                self._cache["long_collateral_contribution"][ticker] = 0
                        else:    # Update the amount
                            if available_cash > 0:
                                self.message.margin_calls[f"long margin call {ticker}"].amount = amount - available_cash
                                self._debt_record[f"long margin call {ticker}"] = amount - available_cash
                                collateral += available_cash
                                self._cache["long_collateral_contribution"][ticker] = available_cash
                                available_cash = 0
                            else:
                                self.message.margin_calls[f"long margin call {ticker}"].amount = amount
                                self._debt_record[f"long margin call {ticker}"] = amount
                                self._cache["long_collateral_contribution"][ticker] = 0
                        new_calls.add(f"long margin call {ticker}")
                    else:    # If margin call is created, we do not increase the collateral.
                        collateral += amount
                        self._cache["long_collateral_contribution"][ticker] = amount
                        available_cash -= amount
                else:
                    self._cache["long_collateral_contribution"][ticker] = 0

        # Remove margin calls that are not called anymore
        # new_call_tickers is a set included in the margin_calls keys.  So we can use set difference
        for call_name in set(self.message.margin_calls.keys()).difference(new_calls):
            if call_name.startswith("missing_funds") or call_name == "short margin call":
                continue
            self.remove_margin_call(call_name)

        return collateral

    def _update_account_collateral(self, timestamp: datetime, security_names: List[str], current_tick_data: np.ndarray,
                                   message: str = "Step Update"):
        """
        Updates the amount of collateral in the account.  This is the amount of money held as collateral and cannot
        be used.  This should be updated at each step because it should be dependent to the current value of the
        assets.  It will also update margin calls at the same time.
        :param timestamp: The date and time of the current step
        :param security_names: An array of the name of each security
        :param current_tick_data: An array of prices of each security for the current step(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :param message: A message to be added to the account history describing why we are doing a collateral update.
        :return: None
        """
        total_cash = self.account.get_total_cash()

        # Step 1: Evaluate short collateral; It also set margin calls if needed
        short_collateral = self._get_short_collateral(total_cash, security_names, current_tick_data)

        # Step 2: Update total_cash  --  Now it is more of a 'available cash'
        total_cash -= short_collateral

        # Step 3: Evaluate long collateral; It also set margin calls if needed
        long_collateral = self._get_long_collateral(total_cash, security_names, current_tick_data)

        # Step 4: Update the collateral so we know how much money is available for trading (Buying power)
        self.account.update_collateral(long_collateral + short_collateral, timestamp, message=message)


    def _liquidate(self, call_amount: float, timestamp: datetime, security_names: List[str],
             tick_data: np.ndarray):
        """
        Liquidate positions to cover margin call.  It starts by short positions.  If there are no short positions or
        are all liquidated and there is still a margin call, it will liquidate long positions.
        If there is not enough money in the portfolio and the account after every positions are liquidated, bankruptcy
        is set to True.
        :param call_amount: The amount of the margin call.
        :param timestamp: The date and time of the current step
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param tick_data:      An array containing prices of each security for the current step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: None
        """
        # For short margin calls, We need to find which position to liquidate to cover call.
        # We liquidate positions that the worth is the closest to the call value.
        # If there is no short positions remaining because the call value is too big, we will liquidate
        # long positions.
        mask = np.zeros(self.portfolio.len_short, dtype=bool)
        liquidated_short = 0
        while call_amount > 1e-8 and liquidated_short < self.portfolio.len_short:
            positions = list(self.portfolio.getShort().values())
            delta = self._get_deltas(call_amount, security_names, tick_data, i=3)
            delta[mask] = -np.inf
            delta_inf = deepcopy(delta)
            delta_inf[delta_inf < 0] = np.inf
            if delta_inf.min() == np.inf:  # No positions are worth enough to payout margin call
                idx = delta.argmax()  # delta are all negatives, so we take the one that is the less negative
                pos = positions[idx]
                # Buy this security
                cash_back = delta[idx] + call_amount
                call_amount -= cash_back
                mask[idx] = True
                self.buy_short(pos.ticker, pos.amount, None, (None, None))
            else:
                pos = positions[delta_inf.argmin()]
                # Buy this security that was sold short
                self.buy_short(pos.ticker, pos.amount, None, (None, None))
                call_amount = 0
            liquidated_short += 1

        # We liquidated all short positions.  We need to liquidate long position to cover.
        if self.portfolio.len_short == liquidated_short and call_amount > 1e-8:
            mask = np.zeros(self.portfolio.len_short, dtype=bool)
            liquidated_long = 0
            while call_amount > 1e-8 and liquidated_long < self.portfolio.len_long:
                positions = list(self.portfolio.getLong().values())
                delta = self._get_deltas(call_amount, security_names, tick_data, short=False, i=3)
                delta[mask] = -np.inf
                delta_inf = deepcopy(delta)
                delta_inf[delta_inf < 0] = np.inf
                if delta_inf.min() == np.inf:  # No positions are worth enough to payout margin call
                    idx = delta.argmax()  # delta are all negatives, so we take the one that is the less negative
                    delta[idx] = -np.inf
                    pos = positions[idx]
                    # Sell this security
                    eq_idx = security_names.index(pos.ticker)
                    price = tuple(tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                    amount = pos.amount
                    cash = self.portfolio.estimateCost(price[0], amount, sell=True) - self._debt_record[pos.ticker]
                    call_amount -= cash
                    mask[idx] = True
                    self.sell_long(pos.ticker, pos.amount, 0, None, (None, None))
                else:
                    pos = positions[delta_inf.argmin()]
                    # Sell this security
                    self.sell_long(pos.ticker, pos.amount, 0, None, (None, None))
                    call_amount = 0

                liquidated_long += 1

            if liquidated_long == self.portfolio.len_long and call_amount > 1e-8:
                self.message.bankruptcy = True
                return

    def _get_deltas(self, reference_amount: float, security_names: List[str], next_tick_data: npt.NDArray[float], short: bool = True,
                    i: int = 0)\
            -> npt.NDArray[float]:
        """
        Get the difference between each short/long position worth and a given price.
        It can be used to compute which position will be the best suited to liquidate to cover a margin call.
        It also include the commission and the collateral contribution in the delta.
        Example:
            # Short
            reference_amount = 10 000, security = [AAPL, MSFT, TSLA], prices = [[100, 101, 99, 100],
                                                                                 [50, 51, 49, 50],
                                                                                 [200, 201, 199, 200]]
            position_size: [100, 50, 150]
            commission: 6.99 (absolute)
            returns: [-7498.2525, -9373.2525, -2498.2525]

        :param reference_amount: The price to compare
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :param i: The index of the price we want to compare with in Open, High Low Close.  (0, 1, 2, 3)
        :param short: If we want to get the delta for short positions.  If False, it will be for long positions.
        :return: Array of the difference between each position worth and the price.  Array shape(n_short_pos, )
        """
        if short:
            delta = []
            for eq in self.portfolio.getShort().values():
                eq_idx = security_names.index(eq.ticker)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                # If positive, it means that the trade will cover the call.
                if self._relative:
                    delta.append(self.min_maintenance_margin_short * (eq.amount * price[i] * self._comm) - reference_amount)
                else:
                    delta.append(self.min_maintenance_margin_short * (eq.amount * price[i] + self._comm) - reference_amount)
        else:
            delta = []
            # Formula to get long collateral:
            # We cannot use cache collateral contribution because we can be looking at the next open data.
            # abs: mk(min_margin - 1) - com(min_margin-1) + debt
            # rel: mk*(2-comm)*(min_margin-1) + debt
            for eq in self.portfolio.getLong().values():
                eq_idx = security_names.index(eq.ticker)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                # If positive, it means that the trade will cover the call.
                if self._relative:
                    amount = eq.amount
                    mk_value = price[i]*amount
                    m = self.min_maintenance_margin - 1    # Precompute the (min_margin - 1) value
                    collateral_contrib = mk_value * (2-self._comm) * m + self._debt_record[eq.ticker]
                    collateral_contrib = collateral_contrib if collateral_contrib > 0 else 0
                    delta.append(mk_value * (2 - self._comm) - self._debt_record[eq.ticker] + collateral_contrib - reference_amount)
                else:
                    amount = eq.amount
                    mk_value = price[i] * amount
                    m = self.min_maintenance_margin - 1  # Precompute the (min_margin - 1) value
                    collateral_contrib = mk_value*m - self._comm * m + self._debt_record[eq.ticker]
                    collateral_contrib = collateral_contrib if collateral_contrib > 0 else 0
                    delta.append(mk_value - self._comm - self._debt_record[eq.ticker] + collateral_contrib - reference_amount)

        return np.array(delta)

    def new_margin_call(self, value: float, message: str = "missing_funds"):
        """
        Create a new margin call and appends it to the other margin calls.  Do not forget to remove money from
        account if necessary.  It also add the value to the debt record.
        :param value: The value that needs to be added to the account to cover margin call
        :return: None
        """
        if message in self.message.margin_calls:
            self.message.margin_calls[message].amount += value
            self._debt_record[message] += value
        else:
            self.message.margin_calls[message] = MarginCall(self.liquidation_delay, value)
            self._debt_record[message] = value

    def remove_margin_call(self, key: str):
        """
        Remove a margin call from the list of margin calls.  It also removed the debt in the debt record.
        :param key: The key of the margin call to remove
        :return: None
        """
        del self.message.margin_calls[key]
        del self._debt_record[key]

    def make_trade(self, order: TradeOrder, security_price: Tuple[float, float, float, float], timestamp: datetime,
                   marginable: bool, shortable: bool) -> bool:
        """
        This method is call to make trades (convert tradeOrders to trade).  Make the trade if price is within limits
        :param order: TradeOrder
        :param security_price: The security price (Open, High, Low, Close)
        :param timestamp: The time where securities will be bought (Usually the next step)
        :param marginable: If the security is marginable.
        :param shortable: If the security is shortable.
        :return: True if the trade was successful (Needs to remove from pending trades) And False otherwise
        :raise RuntimeError: If the trade cannot be executed because of margin or cash issues.
        """
        if order.trade_type == TradeType.BuyLong:
            # Early exit if the security is not marginable and stragegy tried to buy it on margin.
            if order.amount_borrowed > 0 and not marginable:
                return False    # We cannot buy on margin this security
            # We buy if the price is below the low or higher than the high
            # If low limit is None, there is no low limit
            # If high limit is None, there is no high limit
            # If they are both None, we buy at open price (market price)
            # If they are both not None, but the price has moved so much that both conditions are met, we apply
            # Murphy's law and use the worst price (high limit).  (Because we can't check intra steps)
            low, high = order.security_price_limit

            price = self._get_buy_price(low, high, security_price)

            if price is not None:  # We bought
                margin_ratio = order.amount / (order.amount + order.amount_borrowed)
                if margin_ratio < self.min_initial_margin:
                    # The margin on this investment is smaller than the minimal initial margin
                    raise RuntimeError(f"Not enough margin to execute the trade.  "
                                       f"Got: {margin_ratio} but the minimum intial margin is: "
                                       f"{self.min_initial_margin}")

                total = self.portfolio.estimateCost(price, order.amount)

                if total > self.account.get_cash():  # Not enough cash to complete the trade
                    return False
                else:
                    trade = order.convertToTrade(price, timestamp, str(self.n))
                    total = self.portfolio.trade(trade)
                    # Sanity check
                    if total > 0:
                        raise RuntimeError("The total should not be positive when buying long")
                    self.account.withdrawal(-total, timestamp, str(self.n))
                self.n += 1
                return True
            else:
                return False
        elif order.trade_type == TradeType.SellLong:
            # First, check if we have enough shares to sell
            if self.portfolio.getLong().get(order.security) is None:
                return False
            if order.amount + order.amount_borrowed > self.portfolio.getLong().get(order.security).amount:
                return False
            # We sell if the price is below the low (Stop loss) or higher than the high (take profit)
            # If low limit is None, there is no stop loss
            # If high limit is None, there is no take profit and only stop loss.
            # If they are both None, we sell at open price (market price)
            # If they are both not None, but the price has moved so much that both conditions are met, we apply
            # Murphy's law and use the worst price.  (Because we can't check intra steps)
            low, high = order.security_price_limit
            price = self._get_sell_price(low, high, security_price)
            if price is not None:    # We sell

                trade = order.convertToTrade(price, timestamp, str(self.n))
                money = self.portfolio.trade(trade)
                # Remove margin calls if there are any for that position
                if f"long margin call {order.security}" in self.message.margin_calls:
                    self.remove_margin_call(f"long margin call {order.security}")
                # Update account collateral to remove the position's contribution
                if self._cache["long_collateral_contribution"][order.security] > 0:
                    self.account.remove_collateral(self._cache["long_collateral_contribution"][order.security], timestamp,
                                               message=f"Sold long position - {order.security}")
                if money > 0:
                    self.account.deposit(money, timestamp, str(self.n))
                else:
                    # Need to withdraw money from bank account to pay out the bank (Broker)
                    due = -money
                    if due > self.account.get_cash():
                        if self.account.get_cash() > 0:
                            remaining = due - self.account.get_cash()
                            self.account.withdrawal(self.account.get_cash(), timestamp, str(self.n))
                            self.new_margin_call(remaining)
                        else:
                            self.new_margin_call(due)
                    else:
                        self.account.withdrawal(due, timestamp, str(self.n), "Sold margin position at loss")
                self.n += 1
                return True
            else:
                return False

        # For the following two:
        # - We only use the amount of shares borrowed since they are all borrowed.So, the amount of shares has no impact
        # - We need to check if the security is shortable.  If not, we do not execute the trade and let it in pending
        #   trade.  (It could become shortable later)
        elif order.trade_type == TradeType.SellShort:
            # Early exit if the security is not shortable.
            if order.amount_borrowed > 0 and not shortable:
                return False    # We cannot sell short this security
            # We sell if the price is below the low, or higher than the high.
            # If one of both limits are None, there is no limit on that side.
            # If there is no limit, we sell at market price (Open)
            # If there is a limit, we sell at the lowest price of the limit because we cannot look intra-step
            # (Murphy's law)
            low, high = order.security_price_limit
            price = self._get_sell_price(low, high, security_price)

            if price is not None:    # We sell short

                total = self.portfolio.estimateCost(price, order.amount_borrowed, sell=True)

                # Verify if we have enough margin to make the trade
                if self.account.get_cash() / total > 1 + self.min_initial_margin_short:
                    money = self.portfolio.trade(order.convertToTrade(price, timestamp, str(self.n)))
                    self.account.deposit(money, timestamp, transaction_id=str(self.n))
                    if self._relative:
                        self.account.add_collateral((self.min_maintenance_margin + 1) * price * order.amount_borrowed * self._comm,
                                                    timestamp, message=order.security)
                    else:
                        self.account.add_collateral((self.min_maintenance_margin + 1) * (price * order.amount_borrowed + self._comm),
                                                    timestamp, message=order.security)
                    self.n += 1
                    return True
                else:
                    raise RuntimeError(f"Not enough margin to execute the trade.  "
                                       f"Got: {self.account.get_cash() / total} "
                                       f"but the minimum intial margin is: "
                                       f"{1 + self.min_initial_margin}")
            else:
                return False
        elif order.trade_type == TradeType.BuyShort:
            # Early exit if the order doesn't buy any borrowed shares.
            if order.security not in self.portfolio.getShort():
                return False
            # We only need to check the amount borrowed because the TradeOrder object cannot have amount bigger than 0
            # if short.
            if order.amount_borrowed > self.portfolio.getShort()[order.security].amount:
                return False

            # We buy if the price is below the low, or higher than the high (stop loss).
            # If one of both limits are None, there is no limit on that side.
            # If there is no limit, we buy at market price (Open)
            # If there is a limit, we buy at the highest price of the limit because we cannot look intra-step
            # (Murphy's law)
            low, high = order.security_price_limit
            price = self._get_buy_price(low, high, security_price)

            if price is not None:    # We sell short
                total = self.portfolio.estimateCost(price, order.amount_borrowed)
                # We check how much the position influenced the collateral.
                asset_collateral = self._cache["short_collateral_contribution"][order.security]
                # If we have a margin call, we can deduce the amount from the collateral from it since we do not need to
                # hold this collateral anymore.
                if "short margin call" in self.message.margin_calls:
                    short_mc = self.message.margin_calls["short margin call"].amount
                    # If the collateral is bigger than the margin call, we remove the margin call and the remaining
                    # collateral contribution from the account's collateral.
                    # Otherwise, we only subtract the collateral contribution from the margin call.
                    if asset_collateral >= short_mc:
                        asset_collateral -= short_mc
                        mc_time_remaining = self.message.margin_calls["short margin call"].time_remaining
                        self.remove_margin_call("short margin call")
                        # Removing remaining virtual collateral and real collateral from account's collateral
                        self.account.remove_collateral(asset_collateral, timestamp, message="Bought back short position")
                    else:
                        self.message.margin_calls["short margin call"].amount -= asset_collateral
                        self._debt_record["short margin call"] -= asset_collateral
                        self.account.remove_collateral(asset_collateral, timestamp, message="Bought back short position")
                # If we do not have any margin calls, we can simply deduce the collateral contribution from the
                # account's collateral.
                else:
                    self.account.remove_collateral(asset_collateral, timestamp, message="Bought back short position")


                if total > self.account.get_cash():
                    trade = order.convertToTrade(price, timestamp, str(self.n))
                    self.portfolio.trade(trade)
                    self.new_margin_call(total - (self.account.get_cash()), message="missing_funds")
                    # Restore old margin call with updated amount
                    self.account.withdrawal(self.account.get_cash(), timestamp, transaction_id=str(self.n))
                    self.n += 1
                    return True
                else:
                    trade = order.convertToTrade(price, timestamp, str(self.n))
                    self.portfolio.trade(trade)
                    self.account.withdrawal(total, timestamp, transaction_id=str(self.n))
                    self.n += 1
                    return True
            else:
                return False

        else:
            raise RuntimeError(f"Invalid trade type!  Got: {order.trade_type}")

    @staticmethod
    def _get_buy_price(low: Optional[float], high: Optional[float], security_price:Tuple[float, float, float, float]) -> Optional[float]:
        """
        Get the price at which the order will pass according to the limits.  If the price is not within the limits, it
        returns None.
        We buy a stock if the stock price becomes less than the low limit or higher than the high limit.  Because we
        are working in a discrete space, we will buy at limit price if the limit is within the low-high price range.
        We will prioritize the high limit if both limits are met to ensure that we do not obtain optimistic results.
        Finally, if the open price is within the limits, we will buy at open price.  (discontinuity)
        :param low: The low limit
        :param high: The high limit
        :param security_price: The price of the security (Open, High, Low, Close)
        :return: The price at which we should buy the security
        """
        if low is None and high is None:
            price = security_price[0]  # Open
        else:
            if low is None:
                if security_price[0] > high:
                    price = security_price[0]  # We buy at open price
                else:
                    price = high if security_price[1] >= high else None
            elif high is None:
                if security_price[0] < low:
                    price = security_price[0]  # We buy at open price
                else:
                    price = low if security_price[2] <= low else None
            elif security_price[0] >= high:
                price = security_price[0]  # We buy at open price
            elif security_price[0] <= low:
                price = security_price[0]  # We buy at open price
            elif security_price[1] >= high:
                price = high
            elif security_price[2] <= low:
                price = low
            else:
                price = None

        return price

    @staticmethod
    def _get_sell_price(low: Optional[float], high: Optional[float], security_price: Tuple[float, float, float, float])\
            -> Optional[float]:
        """
        Get the price at which the sell order will pass according to the limits.  If the price is not within the limits,
        it returs None.
        We sell a position if the price becomes less than the low limit or higher than the high limit.  Because we are
        working in a discrete space, we will sell at limit price if the limit is within the low-high price range.
        We will prioritize the low limit if both limits are met to ensure that we do not obtain optimistic results.
        Finally, if the open price is within the limits, we will buy at open price.  (discontinuity)
        :param low: The low limit  (If None, it means that there is no stop loss)
        :param high: The high limit (If None, it means that there is no take profit limit)
        :param security_price: The security price (Open, High, Low, Close)
        :return: The sell price if within the limits, None otherwise
        """
        if low is None and high is None:
            price = security_price[0]  # Open
        else:
            if low is None:
                if security_price[0] >= high:
                    price = security_price[0]  # We sell at open price
                else:
                    price = high if security_price[1] >= high else None
            elif high is None:
                if security_price[0] <= low:
                    price = security_price[0]  # We sell at open price
                else:
                    price = low if security_price[2] <= low else None
            elif security_price[0] <= low:
                price = security_price[0]    # We sell at open
            elif security_price[0] >= high:
                price = security_price[0]    # We sell at open
            elif security_price[2] <= low:
                price = low
            elif security_price[1] >= high:
                price = high
            else:
                price = None

        return price

    @staticmethod
    def _isMarginCall(market_value: float, loan: float, min_maintenance_margin: float, transaction_cost: float, rel: bool) -> Tuple[bool, float]:
        """
        Check if there is a margin call (long) and how much is the margin call.
        :param market_value: The current market value of the investment
        :param loan: The value of the loan
        :param min_maintenance_margin: The minimum maintenance margin ratio [0, 1]
        :param transaction_cost: The transaction cost.  Can be absolute or relative to the market value.
                                If relative, must be between [1, 2]  Where 1% would be 1.01 and 100% would be 2.
        :param rel: If the transaction cost is relative to the market value or absolute (in dollars)
        :return: if it is a margin call, the amount of the margin call (if there is so)
        """
        adjusted_market_value = market_value * (2-transaction_cost) if rel else market_value - transaction_cost
        worth = adjusted_market_value - loan
        abs_maintenance_margin = min_maintenance_margin * adjusted_market_value
        if worth <= abs_maintenance_margin:
            return True, abs_maintenance_margin - worth
        else:
            return False, 0

    # @staticmethod
    # def _isShortMarginCall(current_cash: float, market_value: float, min_maintenance_margin: float) -> Tuple[bool, float]:
    #     """
    #     :param current_cash: Current cash in bank account
    #     :param market_value: The total market value of short investments
    #     :param min_maintenance_margin: Minimum maintenance margin [1, inf[
    #     :return: If there is a margin call, amount of the margin call
    #     """
    #     if min_maintenance_margin <= 0 or min_maintenance_margin > 1:
    #         raise ValueError(f"Invalid minimum maintenance margin.  It must be between ]0,1] and got: "
    #                          f"{min_maintenance_margin}")
    #
    #     if current_cash < (1 + min_maintenance_margin) * market_value:
    #         return True, (1 + min_maintenance_margin) * market_value - current_cash
    #     else:
    #         return False, 0


    @staticmethod
    def _findId(s: str, keys: set):
        """
        Find a unique id for a given string
        :param s: A string
        :param keys: The keys in the set or dict where we want to find a unique id
        :return: The unique id
        """
        if s in keys:
            i = 1
            while f"{s}_{i}" in keys:
                i+=1

            return f"{s}_{i}"

        else:
            return s

    def get_state(self) -> dict:
        """
        Get the current state of the broker as a JSONable dictionary (deepcopy
        Note:
            - It doesn't include the account's state.
        :return: The broker's state as a dictionary
        """
        return {
            "portfolio": self.portfolio.get_state(),
            "queued_trade_offers": [offer.export() for offer in self._queued_trade_offers],
            "current_month": self._current_month,
            "last_day": self._last_day,
            "last_step": self._last_step,
            "message": self.message.export(),
            "debt_record": deepcopy(self._debt_record),
            "historical_states": [state.export() for state in self.historical_states],
            "liquidation_delay": self.liquidation_delay,
            "min_initial_margin": self.min_initial_margin,
            "min_maintenance_margin": self.min_maintenance_margin,
            "min_initial_margin_short": self.min_initial_margin_short,
            "min_maintenance_margin_short": self.min_maintenance_margin_short,
            "margin_interest": self.margin_interest,
            "comm": self._comm,
            "relative": self._relative,
            "n": self.n,
            "cache": self._cache,
            "exposure_time": self.exposure_time,
            "current_timestamp": self._current_timestamp,
        }

    @classmethod
    def load_state(cls, data: dict, account: Account):
        """
        Load a broker from a state dictionary
        :param data: The state dictionary
        :param account: The account to which the broker is linked (Loaded by the BacktestResult object)
        :return: The broker
        """
        broker = cls(account)
        broker._last_step = data["last_step"]
        broker._current_timestamp = data["current_timestamp"]
        broker.exposure_time = data["exposure_time"]
        broker._debt_record = deepcopy(data["debt_record"])
        broker.portfolio = Portfolio.load_state(data["portfolio"], broker._debt_record)
        broker._queued_trade_offers = [TradeOrder.load(offer) for offer in data["queued_trade_offers"]]
        broker._current_month = data["current_month"]
        broker._last_day = data["last_day"]
        broker.message = BrokerState.load(data["message"])
        broker.historical_states = [StepState.load(state) for state in data["historical_states"]]
        broker.liquidation_delay = data["liquidation_delay"]
        broker.min_initial_margin = data["min_initial_margin"]
        broker.min_maintenance_margin = data["min_maintenance_margin"]
        broker.min_initial_margin_short = data["min_initial_margin_short"]
        broker.min_maintenance_margin_short = data["min_maintenance_margin_short"]
        broker.margin_interest = data["margin_interest"]
        broker._comm = data["comm"]
        broker._relative = data["relative"]
        broker.n = data["n"]
        broker._cache = data["cache"]
        return broker

    def __eq__(self, other):
        if not isinstance(other, Broker):
            return False
        return (self._comm == other._comm and
                self._relative == other._relative and
                self._current_month == other._current_month and
                self.min_maintenance_margin == other.min_maintenance_margin and
                self.min_initial_margin == other.min_initial_margin and
                self.min_maintenance_margin_short == other.min_maintenance_margin_short and
                self.min_initial_margin_short == other.min_initial_margin_short and
                self.margin_interest == other.margin_interest and
                self.liquidation_delay == other.liquidation_delay and
                self._debt_record == other._debt_record and
                self.message == other.message and
                self._queued_trade_offers == other._queued_trade_offers and
                self.portfolio == other.portfolio and
                self.account == other.account and
                self.n == other.n and
                self._month_interests == other._month_interests and
                self._last_step == other._last_step and
                self._last_day == other._last_day and
                self.historical_states == other.historical_states and
                self._current_timestamp == other._current_timestamp and
                self.exposure_time == other.exposure_time and
                self._cache == other._cache)
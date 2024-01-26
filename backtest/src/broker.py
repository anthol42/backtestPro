import numpy as np
import pandas as pd

from .transaction import Transaction, TransactionType
from .account import Account
from .trade import (BuyLong, BuyShort, SellShort, SellLong, BuyLongOrder, SellLongOrder, BuyShortOrder, SellShortOrder,
                    TradeOrder, TradeType)
from .portfolio import Portfolio, Position
from datetime import timedelta, datetime, date
from typing import Dict, Tuple, List, Union, Optional
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
        return "MARGIN CALL"

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

class StepState:
    """
    Record the state of the broker at each steps for easier strategy debugging
    """
    def __init__(self, timestamp: datetime, worth: float, orders: List[TradeOrder], filled_orders: List[TradeOrder],
                 margin_calls: Dict[str, MarginCall]):
        self.timestamp = timestamp
        self.worth = worth
        self.orders = orders
        self.filled_orders = filled_orders
        self.margin_calls = margin_calls


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
            self._comm = commission
            self._relative = False

        if min_maintenance_margin > 1. or min_maintenance_margin < 0.:
            raise ValueError(f"Minimum maintenance margin must be between 0 and 1.  Got: {min_maintenance_margin}")

        if min_initial_margin > 1. or min_initial_margin < 0.:
            raise ValueError(f"Minimum initial margin must be between 0 and 1.  Got: {min_initial_margin}")

        if min_maintenance_margin >= min_initial_margin:
            raise ValueError(f"Minimum maintenance margin must be smaller or equal than minimum initial margin. ")

        self.min_maintenance_margin = min_maintenance_margin
        self.min_initial_margin = min_initial_margin
        self.margin_interest = margin_interest
        self.min_initial_margin_short = min_initial_margin_short
        self.min_maintenance_margin_short = min_maintenance_margin_short
        self.liquidation_delay = liquidation_delay

        # By-stock record of borrowed money {ticker, borrowed_amount}
        self._debt_record: Dict[str, float] = {}
        self._queued_trade_offers = []
        self.portfolio = Portfolio(self._comm, self._relative, self._debt_record)  # Equities bought with available cash
        self.account = bank_account
        self.n = 0    # Keeps the count of trades.  (To give trades an id)
        self._month_interests = 0    # The current interests of the month.  They will be charged at the end of the month
        self._current_month: Optional[int] = None       # Remember the current month so the broker knows when we change
                                                        # month to charge monthly fees such as interest rates.
        self._last_day: Optional[date] = None           # Remember the last day so the broker knows how long
                                                        # elapsed between the last step and the current step to charge
                                                        # the correct amount of interest
        
        self.message = BrokerState({}, False)

        # Stats
        self._historical_states = List[StepState]


    def buy_long(self, ticker: str, amount: int, expiry: datetime, price_limit: Tuple[float, float] = (None, None),
                 amount_borrowed: int = 0):
        self._queued_trade_offers.append(BuyLongOrder(ticker, price_limit, amount, amount_borrowed, expiry))

    def sell_long(self, ticker: str, amount: int, expiry: datetime, price_limit: Tuple[float, float] = (None, None),
                 amount_borrowed: int = 0):
        self._queued_trade_offers.append(SellLongOrder(ticker, price_limit, amount, amount_borrowed, expiry))

    def sell_short(self, ticker: str, amount: int, expiry: datetime, price_limit: Tuple[float, float] = (None, None),
                 amount_borrowed: int = 0):
        self._queued_trade_offers.append(SellShortOrder(ticker, price_limit, amount, amount_borrowed, expiry))

    def buy_short(self, ticker: str, amount: int, expiry: datetime, price_limit: Tuple[float, float] = (None, None),
                 amount_borrowed: int = 0):
        self._queued_trade_offers.append(BuyShortOrder(ticker, price_limit, amount, amount_borrowed, expiry))

    def tick(self, timestamp: datetime, security_names: List[str], current_tick_data: np.ndarray,
             next_tick_data: np.ndarray, marginables: npt.NDArray[np.bool], dividends: npt.NDArray[np.float32],
             div_freq: List[DividendFrequency]):
        """
        The simulation calls this method after the strategy has run.  It will calculate interests and margin call if
        applicable.  It will do trades that can be done in the trade queue at the next open.
        :param timestamp: The date and time of the current step
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
        :return: None
        """

        # If it is the first step, we need to initialize the portfolio with the current month
        if self._current_month is None:
            self._current_month = timestamp.month

        if self._last_day is None:
            self._last_day = timestamp.date()

        # Step 1: Get the total borrowed money, cash and portfolio worth
        borrowed_money = sum(self._debt_record.values())

        # Cash in dividends
        for ticker, position in self.portfolio.getLong().items():
            ticker_idx = security_names.index(ticker)
            if dividends[ticker_idx] > 0:
                dividend_payout = self.compute_dividend_payout(timestamp, position, div_freq[ticker_idx], dividends[ticker_idx])
                self.account.deposit(dividend_payout,
                                     timestamp, comment=f"Dividends - {ticker}")

        # Evaluate the worth of the portfolio
        worth = self._get_worth(security_names, current_tick_data)

        # Step 2: If the portfolio has borrowed money: we calculate current interests and add them to monthly interests.
        # Interest rates are calculated daily but charged monthly.
        days_elapsed: int = (timestamp.date() - self._last_day).days
        if borrowed_money > 0:
            self._month_interests += days_elapsed * self.margin_interest * borrowed_money / 360

        # Step 3: Update the account collateral after paying debts and interests, if it wasn't paid the previous steps
        missing_funds_calls = [ticker for ticker in self.message.margin_calls if ticker.startswith("missing_funds")]
        for miss_fund_call in missing_funds_calls:
            if self.account.get_cash() > self.message.margin_calls[miss_fund_call].amount:
                self.remove_margin_call(miss_fund_call)
                self.account.withdrawal(self.message.margin_calls[miss_fund_call].amount, timestamp,
                                        comment="Missing funds")
        self._update_account_collateral(timestamp, security_names, current_tick_data)

        # Step 4: Execute trades that can be executed
        filled_orders = []
        for order in self._queued_trade_offers:
            if order.expiry <= timestamp:
                continue
            eq_idx = security_names.index(order.ticker)

            result = self.make_trade(order, next_tick_data[eq_idx], timestamp, marginables[eq_idx, 0], marginables[eq_idx, 1])
            if result:
                filled_orders.append(order)

        for order in filled_orders:
            self._queued_trade_offers.remove(order)

        # Step 5: If there is borrowed money, reduce the delay of margin calls.  (It is normal that newly initialized
        # margin call's delay will already be reduced.  It is took into account in the liquidation process)
        if borrowed_money > 0:
            # Decrement margin call delay until liquidation.  If it reaches -1, the position is liquidated
            for key in self.message.margin_calls:
                self.message.margin_calls[key] -= 1

        # Step 6: Charge interests if it's the first day of the month
        # Interest are deducted from account.  If there is not enough money in the account to payout interests,
        # the account, we crrate a new margin call for missing funds and interests will be charged on these because.
        if timestamp.month != self._current_month:
            self._current_month = timestamp.month
            if self.account.get_cash() > self._month_interests:
                self.account.withdrawal(self._month_interests, timestamp, comment="Interest payment")
                self._month_interests = 0
            else:
                # Put interests as debt and as margin call that the user needs to pay
                self.new_margin_call(self._month_interests)
                self._month_interests = 0

        # Step 6: Liquidate expired margin calls
        pos_to_liquidate = [ticker for ticker in self.message.margin_calls if self.message.margin_calls[ticker] == -1]
        for ticker in pos_to_liquidate:
            if ticker.startswith("missing_funds") or ticker == "short margin call":
                continue
            eq_idx = security_names.index(ticker)
            price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
            # We sell at this price if it is long
            long, short = self.portfolio[ticker]  # Can be long AND short even though it does not make that much sense
            if long is not None:
                order = SellLongOrder(ticker, (None, None), long.amount, long.amount_borrowed, None)
                self.make_trade(order, price, timestamp)

        if self.message.margin_calls["short margin call"].time_remaining == -1:
            self._liquidate(self.message.margin_calls["short margin call"].amount, timestamp, security_names,
                            next_tick_data)
        missing_funds_call = [ticker for ticker in self.message.margin_calls if ticker.startswith("missing_funds")]
        for miss_fund_call in missing_funds_call:
            if self.message.margin_calls[miss_fund_call].time_remaining == -1:
                self._liquidate(self.message.margin_calls[miss_fund_call].amount, timestamp, security_names,
                                next_tick_data)

        # Step 6: Save states
        self._historical_states.append(
            StepState(timestamp, worth, self._queued_trade_offers, filled_orders, self.message.margin_calls)
        )

    def compute_dividend_payout(self, timestamp: datetime, position: Position,
                                div_freq: DividendFrequency, dividends: np.ndarray) -> float:
        """
        Compute the dividend payout of a long position
        :param timestamp: The current timestamp
        :param position: The position to calculate dividend payout
        :param div_freq: The frequency that the security is paying dividends.
        :param dividends: The actual dividends payout for 1 share and for the period in div_freq.
        :return: Dividend payout
        """
        hold_days = (timestamp - position.last_dividends_dt).total_seconds() / 3600 / 24
        if div_freq == DividendFrequency.MONTHLY:
            hold_ratio = hold_days / 30
        elif div_freq == DividendFrequency.QUARTERLY:
            hold_ratio = hold_days / 90
        elif div_freq == DividendFrequency.BIANNUALLY:
            hold_ratio = hold_days / 180
        else:  # DividendFrequency.YEARLY
            hold_ratio = hold_days / 365

        if hold_ratio > 1:
            hold_ratio = 1
        return dividends * (position.amount + position.amount_borrowed) * hold_ratio


    def _get_short_collateral(self, available_cash: float, security_names: List[str],
                              current_tick_data: npt.NDArray[float]) -> float:
        """
        Get the total collateral of short positions  It will also update margin calls at the same time.
        :param available_cash: The available cash in the account.  (Total - reserved for collateral)
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the current step shape(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: The total collateral of short positions
        """
        collateral = 0.
        for ticker in self.portfolio.getShort():
            eq_idx = security_names.index(ticker)
            price = tuple(current_tick_data[eq_idx].tolist()) # (Open, High, Low, Close)
            mk_value = self.portfolio.getShort()[ticker].amount_borrowed * price[3] * (2 - self._comm)
            collateral += (1 + self.min_maintenance_margin_short) * mk_value

        if available_cash - collateral < 0:
            if "short margin call" in self.message.margin_calls:
                self.message.margin_calls["short margin call"].amount = collateral - available_cash
            else:
                self.new_margin_call(collateral - available_cash, "short margin call")
        else:
            self.remove_margin_call("short margin call")

        return collateral

    def _get_worth(self, security_names: List[str], current_tick_data: np.ndarray) -> float:
        """
        Get the worth of the portfolio
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
            worth += (position.amount * close +
                      position.amount_borrowed * close - debt)

        for ticker, position in self.portfolio.getShort().items():
            eq_idx = security_names.index(ticker)
            price = tuple(current_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
            close = price[3]
            worth -= position.amount_borrowed * close

        return worth
    def _get_long_collateral(self, available_cash: float, security_names: List[str],
                             current_tick_data: npt.NDArray[float]) -> float:
        """
        Get the total collateral of long positions (Only the one having a margin call impact the collateral requirement)
        It will also update margin calls at the same time ( remove the ones that aren't called anymore and create the
        new ones.)
        :param available_cash: The available cash in the account.  (Total - reserved for collateral)
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param current_tick_data: An array containing prices of each security for the current step shape(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: the collateral of long positions
        """
        new_call_tickers = set()
        collateral = 0.
        for ticker in self._debt_record:
            long, short = self.portfolio[ticker]

            eq_idx = security_names.index(ticker)
            if long is not None:
                if long.ticker in self.message.margin_calls:
                    continue
                # Verify if the stock is in margin call for long
                is_margin_call, amount = self._isMarginCall(long.amount * current_tick_data[eq_idx, -1],
                                                            self._debt_record[ticker],
                                                            self.min_maintenance_margin)
                if is_margin_call:
                    # Check if true margin call compared with account balance
                    if amount > available_cash:
                        # Only adds it if not in margin call already.  (to not reset the delay)
                        if ticker not in self.message.margin_calls:
                            self.new_margin_call(amount - available_cash, ticker)
                        new_call_tickers.add(ticker)
                available_cash -= amount
                collateral += amount
        # Remove margin calls that are not called anymore
        # new_call_tickers is a set included in the margin_calls keys.  So we can use set difference
        for ticker in set(self.message.margin_calls.keys()).difference(new_call_tickers):
            if ticker.startswith("missing_funds") or ticker == "short margin call":
                continue
            self.remove_margin_call(ticker)

        return collateral

    def _update_account_collateral(self, timestamp: datetime, security_names: List[str], current_tick_data: np.ndarray):
        """
        Updates the amount of collateral in the account.  This is the amount of money held as collateral and cannot
        be used.  This should be updated at each step because it should be dependent to the current value of the
        assets.  It will also update margin calls at the same time.
        :param timestamp: The date and time of the current step
        :param security_names: An array of the name of each security
        :param current_tick_data: An array of prices of each security for the current step(n_securities, 4)
                                  The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: None
        """
        total_cash = self.account.get_cash()

        # Step 1: Evaluate short collateral; It also set margin calls if needed
        short_collateral = self._get_short_collateral(total_cash, security_names, current_tick_data)

        # Step 2: Update total_cash  --  Now it is more of a 'available cash'
        total_cash -= short_collateral

        # Step 3: Evaluate long collateral; It also set margin calls if needed
        long_collateral = self._get_long_collateral(total_cash, security_names, current_tick_data)

        # Step 4: Update the collateral so we know how much money is available for trading (Buying power)
        self.account.update_collateral(long_collateral + short_collateral, timestamp)


    def _liquidate(self, call_amount: float, timestamp: datetime, security_names: List[str],
             next_tick_data: np.ndarray):
        """
        Liquidate positions to cover margin call.  It starts by short positions.  If there are no short positions or
        are all liquidated and there is still a margin call, it will liquidate long positions.
        If there is not enough money in the portfolio and the account after every positions are liquidated, bankruptcy
        is set to True.
        :param call_amount: The amount of the margin call.
        :param timestamp: The date and time of the current step
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: None
        """
        # For short margin calls, We need to find which position to liquidate to cover call.
        # We liquidate positions that the worth is the closest to the call value.
        # If there is no short positions remaining because the call value is too big, we will liquidate
        # long positions.
        while call_amount > 0 and len(self.portfolio.getShort()) > 0:
            positions = list(self.portfolio.getShort().values())
            delta = self._get_deltas(call_amount, security_names, next_tick_data)
            delta_inf = deepcopy(delta)
            delta_inf[delta_inf < 0] = np.inf
            if delta_inf.min() == np.inf:  # No positions are worth enough to payout margin call
                idx = delta.argmax()  # delta are all negatives, so we take the one that is the less negative
                delta[idx] = -np.inf
                eq = positions[idx]
                # Buy this security
                eq_idx = security_names.index(eq)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                if self._relative:
                    cash = eq.amount_borrowed * price[0] * (2 - self._comm)
                else:
                    cash = eq.amount_borrowed * price[0] - self._comm
                call_amount -= cash
                order = BuyShortOrder(eq.ticker, (None, None), eq.amount, eq.amount_borrowed, None)
                self.make_trade(order, price, timestamp)
            else:
                eq = positions[delta_inf.argmin()]
                # Buy this security
                eq_idx = security_names.index(eq)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                order = BuyShortOrder(eq.ticker, (None, None), eq.amount, eq.amount_borrowed, None)
                self.make_trade(order, price, timestamp)
                call_amount = 0

        # We liquidated all short positions.  We need to liquidate long position to cover.
        if len(self.portfolio.getShort()) == 0 and call_amount > 0:
            while call_amount > 0 and len(self.portfolio.getLong()) > 0:
                positions = list(self.portfolio.getLong().values())
                delta = self._get_deltas(call_amount, security_names, next_tick_data, short=False)
                delta_inf = deepcopy(delta)
                delta_inf[delta_inf < 0] = np.inf
                if delta_inf.min() == np.inf:  # No positions are worth enough to payout margin call
                    idx = delta.argmax()  # delta are all negatives, so we take the one that is the less negative
                    delta[idx] = -np.inf
                    eq = positions[idx]
                    # Buy this security
                    eq_idx = security_names.index(eq)
                    price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                    if self._relative:
                        amount = eq.amount + eq.amount_borrowed
                        cash = amount * price[0] * (2 - self._comm) - self._debt_record[eq.ticker]
                    else:
                        amount = eq.amount + eq.amount_borrowed
                        cash = amount * price[0] - self._comm - self._debt_record[eq.ticker]
                    call_amount -= cash
                    order = SellLongOrder(eq.ticker, (None, None), eq.amount, eq.amount_borrowed, None)
                    self.make_trade(order, price, timestamp)
                else:
                    eq = positions[delta_inf.argmin()]
                    # Buy this security
                    eq_idx = security_names.index(eq)
                    price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                    order = SellLongOrder(eq.ticker, (None, None), eq.amount, eq.amount_borrowed, None)
                    self.make_trade(order, price, timestamp)
                    call_amount = 0

            if len(self.portfolio.getLong()) == 0 and call_amount > 0:
                self.message.bankruptcy = True
                return

    def _get_deltas(self, call_amount: float, security_names: List[str], next_tick_data: npt.NDArray[float], short: bool = True)\
            -> npt.NDArray[float]:
        """
        Get the difference between each short/long position worth and a given price.
        :param call_amount: The price to compare
        :param security_names: A list of all securities where the index of each ticker is the index of the data of the
                               corresponding security in the 'next_tick_data' parameter along the first axis (axis=0).
        :param next_tick_data: An array containing prices of each security for the next step shape(n_securities, 4)
                               The 4 columns of the array are: Open, High, Low, Close of the next step.
        :return: Array of the difference between each position worth and the price.  Array shape(n_short_pos, )
        """
        if short:
            delta = []
            for eq in self.portfolio.getShort().values():
                eq_idx = security_names.index(eq.ticker)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                # If positive, it means that the trade will cover the call.
                if self._relative:
                    delta.append(eq.amount_borrowed * price[0] * (2 - self._comm) - call_amount)
                else:
                    delta.append(eq.amount_borrowed * price[0] - self._comm - call_amount)
        else:
            delta = []
            for eq in self.portfolio.getLong().values():
                eq_idx = security_names.index(eq.ticker)
                price = tuple(next_tick_data[eq_idx].tolist())  # (Open, High, Low, Close)
                # If positive, it means that the trade will cover the call.
                if self._relative:
                    amount = eq.amount + eq.amount_borrowed
                    delta.append(amount * price[0] * (2 - self._comm) - self._debt_record[eq.ticker] - call_amount)
                else:
                    amount = eq.amount + eq.amount_borrowed
                    delta.append(amount * price[0] - self._comm - self._debt_record[eq.ticker] - call_amount)

        return np.array(delta)

    def new_margin_call(self, value: float, message: str = "missing_funds"):
        """
        Create a new margin call and appends it to the other margin calls.  Do not forget to remove money from
        account if necessary.
        :param value: The value that needs to be added to the account to cover margin call
        :return: None
        """
        new_key = self._findId(message, set(self.message.margin_calls.keys()))
        self.message.margin_calls[new_key] = MarginCall(self.liquidation_delay, value)
        self._debt_record[new_key] = value

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
        This method is call to make trades (convert tradeOrders to trade).  Make the trade if price is in limit
        :param order: TradeOrder
        :param security_price: The security price (Open, High, Low, Close)
        :param timestamp: The time where securities will be bought (Usually the next step)
        :param marginable: If the security is marginable.
        :param shortable: If the security is shortable.
        :return: True if the trade was successful (Needs to remove from pending trades) And False otherwise
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
            if low is None and high is None:
                price = security_price[0]    # Open
            else:
                if low is None:
                    if security_price[1] > high:
                        price = high
                elif high is None:
                    if security_price[2] < low:
                        price = low
                elif security_price[1] > high:
                    price = high
                elif security_price[2] < low:
                    price = low
                else:
                    price = None

            if price is not None:  # We bought
                margin_ratio = order.amount / (order.amount + order.amount_borrowed)
                if margin_ratio > self.min_initial_margin:
                    # The margin on this investment is smaller than the minimal initial margin
                    raise RuntimeError(f"Not enough margin to execute the trade.  "
                                       f"Got: {margin_ratio} but the minimum intial margin is: "
                                       f"{self.min_initial_margin}")
                if self._relative:
                    total = order.amount * price * self._comm
                else:
                    total = order.amount * price + self._comm

                if total > self.account.get_cash():  # Not enough cash to complete the trade
                    return False
                else:
                    trade = order.convertToTrade(price, timestamp, str(self.n))
                    self.portfolio.trade(trade)
                    self.account.withdrawal(total, timestamp, str(self.n))
                    if order.security in self._debt_record:
                        self._debt_record[order.security] += order.amount_borrowed * price * self._comm
                    else:
                        self._debt_record[order.security] = order.amount_borrowed * price * self._comm
                self.n += 1
                return True
            else:
                return False
        elif order.trade_type == TradeType.SellLong:
            # We sell if the price is below the low (Stop loss) or higher than the high (take profit)
            # If low limit is None, there is no stop loss
            # If high limit is None, there is no take profit and only stop loss.
            # If they are both None, we sell at open price (market price)
            # If they are both not None, but the price has moved so much that both conditions are met, we apply
            # Murphy's law and use the worst price.  (Because we can't check intra steps)
            low, high = order.security_price_limit
            if low is None and high is None:
                price = security_price[0]    # Open
            else:
                if low is None:
                    if security_price[1] > high:
                        price = high
                elif high is None:
                    if security_price[2] < low:
                        price = low
                elif security_price[2] < low:
                    price = low
                elif security_price[1] > high:
                    price = high
                else:
                    price = None
            if price is not None:    # We sell
                if self._relative:
                    total = (order.amount + order.amount_borrowed) * price * (2 - self._comm)
                else:
                    total = (order.amount + order.amount_borrowed) * price - self._comm

                trade = order.convertToTrade(price, timestamp, str(self.n))
                self.portfolio.trade(trade)
                money = total - self._debt_record[order.security]
                del self._debt_record[order.security]
                if money > 0:
                    self.account.deposit(money, timestamp)
                else:
                    # Need to withdraw money from bank account to pay out the bank (Broker)
                    due = -money
                    if due > self.account.get_cash():
                        self.new_margin_call(due)
                    else:
                        self.account.withdrawal(due, timestamp, "Sold margin position at loss")
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
            if low is None and high is None:
                price = security_price[0]  # Open
            else:
                if low is None:
                    if security_price[1] > high:
                        price = high
                elif high is None:
                    if security_price[2] < low:
                        price = low
                elif security_price[2] < low:
                    price = low
                elif security_price[1] > high:
                    price = high
                else:
                    price = None

            if price is not None:    # We sell short
                if self._relative:
                    total = order.amount_borrowed * price * (2 - self._comm)
                else:
                    total = order.amount_borrowed * price - self._comm

                # Verify if we have enough margin to make the trade
                if self.min_initial_margin_short * total > self.account.get_cash():
                    self.account.deposit(total, timestamp, transaction_id=str(self.n))
                    self.account.add_collateral((self.min_maintenance_margin + 1) * total, timestamp, message=order.security)
                    self.portfolio.trade(order.convertToTrade(price, timestamp, str(self.n)))
                    self.n += 1
                    return True
                else:
                    return False
            else:
                return False
        elif order.trade_type == TradeType.BuyShort:
            # Early exit if the order doesn't buy any borrowed shares.
            if order.amount_borrowed > 0:
                return False  # We cannot buy short this security
            # We buy if the price is below the low, or higher than the high (stop loss).
            # If one of both limits are None, there is no limit on that side.
            # If there is no limit, we buy at market price (Open)
            # If there is a limit, we buy at the highest price of the limit because we cannot look intra-step
            # (Murphy's law)
            low, high = order.security_price_limit
            if low is None and high is None:
                price = security_price[0]  # Open
            else:
                if low is None:
                    if security_price[1] > high:
                        price = high
                elif high is None:
                    if security_price[2] < low:
                        price = low
                elif security_price[1] > high:
                    price = high
                elif security_price[2] < low:
                    price = low
                else:
                    price = None

            if price is not None:    # We sell short
                if self._relative:
                    total = order.amount_borrowed * price * self._comm
                else:
                    total = order.amount_borrowed * price + self._comm

                if total > self.account.get_cash():
                    self.new_margin_call(total - self.account.get_total_cash(), message="missing_funds")
                    trade = order.convertToTrade(price, timestamp, str(self.n))
                    self.portfolio.trade(trade)
                    self.account.withdrawal(total - self.account.get_total_cash(), timestamp, transaction_id=str(self.n))
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

    def _isMarginCall(self, market_value: float, loan: float, min_maintenance_margin: float) -> Tuple[bool, float]:
        """
        Check if there is a margin call (long) and how much is the margin call.  If there is enough cash in account,
        the margin call is ignored.
        :param market_value: The current market value of the investment
        :param loan: The value of the loan
        :param min_maintenance_margin: The minimum maintenance margin ratio [0, 1]
        :return: if it is a margin call, the amount of the margin call (if there is so)
        """
        if min_maintenance_margin <= 0 or min_maintenance_margin > 1:
            raise ValueError(f"Invalid minimum maintenance margin.  It must be between ]0,1] and got: "
                             f"{min_maintenance_margin}")
        equity = market_value - loan
        abs_maintenance_margin = min_maintenance_margin * market_value
        if equity <= abs_maintenance_margin:
            # call_amount = (abs_maintenance_margin - equity)
            # if call_amount > self.account.get_cash():
            #     return True, call_amount - self.account.get_cash()
            # else:
            #     return False, 0
            return True, abs_maintenance_margin - equity
        else:
            return False, 0

    @staticmethod
    def _isShortMarginCall(current_cash: float, market_value: float, min_maintenance_margin: float) -> Tuple[bool, float]:
        """
        :param current_cash: Current cash in bank account
        :param market_value: The total market value of short investments
        :param min_maintenance_margin: Minimum maintenance margin [1, inf[
        :return: If there is a margin call, amount of the margin call
        """
        if min_maintenance_margin <= 0 or min_maintenance_margin > 1:
            raise ValueError(f"Invalid minimum maintenance margin.  It must be between ]0,1] and got: "
                             f"{min_maintenance_margin}")

        if current_cash < (1 + min_maintenance_margin) * market_value:
            return True, (1 + min_maintenance_margin) * market_value - current_cash
        else:
            return False, 0


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
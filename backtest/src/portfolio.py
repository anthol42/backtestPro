from .trade import Trade, BuyLong, BuyShort, SellLong, SellShort
from copy import deepcopy
from typing import Dict, Tuple, List, Union
from datetime import datetime, timedelta

class Position:
    """
    Data class holding info about a position
    """
    def __init__(self, ticker: str, amount: int,  amount_borrowed: int, long: bool, average_price: float, average_filled_price: datetime):
        self.ticker = ticker
        self.amount = amount
        self.amount_borrowed = amount_borrowed
        self.average_price = average_price
        self.on_margin = amount_borrowed > 0
        self.long = long
        self.average_filled_price = average_filled_price

    @property
    def worth(self):
        return self.average_price * (self.amount + self.amount_borrowed)

    def __str__(self):
        mrg = "MARGIN" if self.on_margin else ""
        lg = "LONG" if self.long else "SHORT"
        return f"EQUITY: {self.amount + self.amount_borrowed}x{self.ticker} {round(self.average_price, 2)}$ {mrg} {lg}"

    def get_total(self):
        """
        :return: Total number of shares
        """
        return self.amount + self.amount_borrowed

    def __repr__(self):
        return f"EQUITY: {self.ticker}"

    def __add__(self, other):
        if isinstance(other, int):
            self.amount += other
        elif isinstance(other, Position):
            current_amount = self.amount + self.amount_borrowed
            other_amount = other.amount + other.amount_borrowed
            total_amount = current_amount + other_amount
            self.average_price = ((current_amount / total_amount) * self.average_price +
                                  (other_amount / total_amount) * other.average_price)
            self.amount += other.amount
            self.amount_borrowed += other.amount_borrowed
            self.average_filled_price = (self.average_filled_price +
                                         timedelta(seconds=(other.average_filled_price -
                                                            self.average_filled_price).total_seconds() / 2))

        elif isinstance(other, Trade):
            current_amount = self.amount + self.amount_borrowed
            other_amount = other.amount + other.amount_borrowed
            total_amount = current_amount + other_amount
            self.average_price = ((current_amount / total_amount) * self.average_price +
                                  (other_amount / total_amount) * other.security_price)
            self.amount += other.amount
            self.amount_borrowed += other.amount_borrowed
        else:
            raise NotImplementedError(f"Addition not implemented for type {type(other)}")

    def __sub__(self, other):
        if isinstance(other, int):
            self.amount -= other
        elif isinstance(other, Position):
            self.amount -= other.amount
            self.amount_borrowed -= other.amount_borrowed
        elif isinstance(other, Trade):
            self.amount -= other.amount
            self.amount_borrowed -= other.amount_borrowed
        else:
            raise NotImplementedError(f"Addition not implemented for type {type(other)}")


class TradeStats:
    """
    This class will hold the stats for a given trade.  It will be used to calculate the stats for the whole portfolio
    """
    def __init__(self, trade: Trade, duration: timedelta, profit: float, rel_profit: float):
        """
        :param trade: The trade object (SellLong or BuyShort)
        :param duration: The duration of the trade
        :param profit: The profit made on the trade
        :param rel_profit: The profit made on the trade relative to the amount invested
        """
        self.trade = trade
        self.duration = duration
        self.profit = profit
        self.rel_profit = rel_profit

class Portfolio:
    """
    This  class will have two sub portfolios: long and short.  When 'trade' is called, it will add or remove in the
    appropriate portfolio the security.  It will also remember each trades (in a list) to recover the state at each
    days (For debugging purpose).
    """
    def __init__(self , transaction_cost: float = 10., transaction_relative: bool = False, debt_record: Dict[str, float] = {}):
        """
        :param transaction_cost: The cost of a transaction (buy or sell) in $ or in %
        :param transaction_relative: Whether the transaction cost is in percentage relative to transaction cost or fix price
        :param debt_record: The amount of debt used to buy securities: {security: amount}.  Passed by reference.
        """
        # Keys will be security (ticker) and the value will be Equity data object
        self._long: Dict[str, Position] = {}
        self._short: Dict[str, Position] = {}
        self._trades: List[Union[TradeStats, Trade]] = []    # TradeStats when closing trade and Trade when opening one.
        self._transaction_cost = transaction_cost
        self._relative = transaction_relative
        self._debt_record: Dict[str, float] = debt_record

    def trade(self, trade: Trade):
        """
        Make a trade and add it to the portfolio.
        :param trade: Can be BuyLong, SellLong, SellShort, BuyShort
        :return: None
        """
        if isinstance(trade, BuyLong):
            if trade.security in self._long:
                self._long[trade.security] += trade
            else:
                self._long[trade.security] = Position(trade.security, trade.amount, trade.amount_borrowed, True,
                                                    trade.security_price)
            self._trades.append(trade)
        elif isinstance(trade, SellLong):
            if trade.security not in self._long:
                raise RuntimeError("Cannot sell Long if the security is not acquired.")
            elif self._long[trade.security].get_total() < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot sell Long more securities than the portfolio has")
            else:
                duration = trade.timestamp - self._long[trade.security].average_filled_price
                average_buy_price = self._long[trade.security].average_price
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               *self.getLongProfit(average_buy_price,
                                                   trade.security_price,
                                                   trade.amount + trade.amount_borrowed,
                                                   self._debt_record[trade.security])))
                self._long[trade.security] -= trade
                if self._long[trade.security].amount == 0 and self._long[trade.security].amount_borrowed == 0:
                    del self._long[trade.security]

        elif isinstance(trade, SellShort):
            if trade.security in self._short:
                self._short[trade.security] += trade.amount_borrowed
            else:
                self._short[trade.security] = Position(trade.security, 0, trade.amount_borrowed, False,
                                                     trade.security_price)
            self._trades.append(trade)
        elif isinstance(trade, BuyShort):
            if trade.security not in self._short:
                raise RuntimeError("Cannot buy short if the security has not been sold short.")
            elif self._short[trade.security].get_total() < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot buy short more securities than the portfolio has sold short.")
            else:
                duration = trade.timestamp - self._long[trade.security].average_filled_price
                average_sell_price = self._short[trade.security].average_price
                trade.amount = 0    # Just to make sure
                self._short[trade.security] -= trade
                if self._short[trade.security].amount_borrowed == 0:
                    del self._short[trade.security]
                self._trades.append(
                    TradeStats(trade,
                               duration,
                               *self.getLongProfit(average_sell_price,trade.security_price, trade.amount_borrowed)))

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
            return gain, gain / average_sell_price
        else:
            gain = (average_buy_price*qty - average_sell_price*qty) - self._transaction_cost
            return gain, gain / average_sell_price

    def getLongProfit(self, average_buy_price: float, average_sell_price: float, qty: int, debt) -> Tuple[float, float]:
        """
        Calculate the profit made on a long trade
        :param average_buy_price: The price at which the security was bought
        :param average_sell_price: The price at which the security was sold
        :param qty: The number of shares in the trade
        :param debt: The amount of debt used to buy the security
        :return: The profit (positive) or loss (negative) made on the trade, relative profit
        """
        if self._relative:
            gain = (average_sell_price*qty - average_buy_price*qty) * (1 - self._transaction_cost) - debt
            return gain, gain / average_buy_price
        else:
            gain = (average_sell_price*qty - average_buy_price*qty) - self._transaction_cost - debt
            return gain, gain / average_buy_price

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
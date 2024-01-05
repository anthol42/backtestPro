from .trade import Trade, BuyLong, BuyShort, SellLong, SellShort
from copy import deepcopy
from typing import Dict, Tuple

class Equity:
    """
    Data class holding info about a position
    """
    def __init__(self, ticker: str, amount: int,  amount_borrowed: int, long: bool, average_price: float):
        self.ticker = ticker
        self.amount = amount
        self.amount_borrowed = amount_borrowed
        self.average_price = average_price
        self.on_margin = amount_borrowed > 0
        self.long = long

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
        elif isinstance(other, Equity):
            current_amount = self.amount + self.amount_borrowed
            other_amount = other.amount + other.amount_borrowed
            total_amount = current_amount + other_amount
            self.average_price = ((current_amount / total_amount) * self.average_price +
                                  (other_amount / total_amount) * other.average_price)
            self.amount += other.amount
            self.amount_borrowed += other.amount_borrowed

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
        elif isinstance(other, Equity):
            self.amount -= other.amount
            self.amount_borrowed -= other.amount_borrowed
        elif isinstance(other, Trade):
            self.amount -= other.amount
            self.amount_borrowed -= other.amount_borrowed
        else:
            raise NotImplementedError(f"Addition not implemented for type {type(other)}")


class Portfolio:
    """
    This  class will have two sub portfolios: long and short.  When 'trade' is called, it will add or remove in the
    appropriate portfolio the security.  It will also remember each trades (in a list) to recover the state at each
    days (For debugging purpose).
    """
    def __init__(self):
        # Keys will be security (ticker) and the value will be Equity data object
        self._long: Dict[str, Equity] = {}
        self._short: Dict[str, Equity] = {}
        self._trades = []

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
                self._long[trade.security] = Equity(trade.security, trade.amount, trade.amount_borrowed, True,
                                                    trade.security_price)
        elif isinstance(trade, SellLong):
            if trade.security not in self._long:
                raise RuntimeError("Cannot sell Long if the security is not acquired.")
            elif self._long[trade.security].get_total() < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot sell Long more securities than the portfolio has")
            else:
                self._long[trade.security] -= trade
                if self._long[trade.security].amount == 0 and self._long[trade.security].amount_borrowed == 0:
                    del self._long[trade.security]
        elif isinstance(trade, SellShort):
            if trade.security in self._short:
                self._short[trade.security] += trade.amount
            else:
                self._short[trade.security] = Equity(trade.security, 0, trade.amount_borrowed, False,
                                                     trade.security_price)
        elif isinstance(trade, BuyShort):
            if trade.security not in self._short:
                raise RuntimeError("Cannot buy short if the security has not been sold short.")
            elif self._short[trade.security].get_total() < trade.amount + trade.amount_borrowed:
                raise RuntimeError("Cannot buy short more securities than the portfolio has sold short.")
            else:
                self._short[trade.security] -= trade
                if self._long[trade.security].amount == 0 and self._long[trade.security].amount_borrowed == 0:
                    del self._long[trade.security]
        self._trades.append(trade)

    def getShort(self) -> Dict[str, Equity]:
        """
        To get what securities are sold short to later calculate interests
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._short)
    def getLong(self) -> Dict[str, Equity]:
        """
        To get what securities are bought long to later calculate interests (When using margin)
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._long)

    def __getitem__(self, item: str) -> Tuple[Equity, Equity]:
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
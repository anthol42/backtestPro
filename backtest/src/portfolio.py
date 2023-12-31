from .trade import Trade, BuyLong, BuyShort, SellLong, SellShort
from copy import deepcopy
from typing import Dict

class Portfolio:
    """
    This  class will have two sub portfolios: long and short.  When 'trade' is called, it will add or remove in the
    appropriate portfolio the security.  It will also remember each trades (in a list) to recover the state at each
    days (For debugging purpose).
    """
    def __init__(self):
        self._long = {}    # Keys will be security (ticker) and the value will be number of stocks
        self._short = {}   # Keys will be security (ticker) and the value will be number of stocks
        self._trades = []

    def trade(self, trade: Trade):
        """
        Make a trade and add it to the portfolio.
        :param trade: Can be BuyLong, SellLong, SellShort, BuyShort
        :return: None
        """
        if isinstance(trade, BuyLong):
            if trade.security in self._long:
                self._long[trade.security] += trade.amount
            else:
                self._long[trade.security] = trade.amount
        elif isinstance(trade, SellLong):
            if trade.security not in self._long:
                raise RuntimeError("Cannot sell Long if the security is not acquired.")
            elif self._long[trade.security] < trade.amount:
                raise RuntimeError("Cannot sell Long more securities than the portfolio has")
            else:
                self._long[trade.security] -= trade.amount
        elif isinstance(trade, SellShort):
            if trade.security in self._short:
                self._short[trade.security] += trade.amount
            else:
                self._short[trade.security] = trade.amount
        elif isinstance(trade, BuyShort):
            if trade.security not in self._short:
                raise RuntimeError("Cannot buy short if the security has not been sold short.")
            elif self._short[trade.security] < trade.amount:
                raise RuntimeError("Cannot buy short more securities than the portfolio has sold short.")
            else:
                self._short[trade.security] -= trade.amount
        self._trades.append(trade)

    def getShort(self) -> Dict[str, int]:
        """
        To get what securities are sold short to later calculate interests
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._short)
    def getLong(self) -> Dict[str, int]:
        """
        To get what securities are bought long to later calculate interests (When using margin)
        :return: Dictionary[security, amount]
        """
        return deepcopy(self._long)
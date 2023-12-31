from .transaction import Transaction, TransactionType
from .trade import BuyLong, BuyShort, SellShort, SellLong
from .portfolio import Portfolio

class Broker:
    def __init__(self, buy_on_close: bool = False, commission: float = None,
                 relative_commission: float = None):
        self._bonc = buy_on_close
        if commission is not None and relative_commission is not None:
            raise ValueError("Must choose between relative commission or absolute commission!")
        if commission is None and relative_commission is None:
            commission = 0

        if relative_commission is not None:
            self._comm = relative_commission
            self._relative = True
        else:
            self._comm = commission
            self._relative = False

        self.n = 0
        self.queued_trade_offers = []
        self.portfolio = Portfolio()

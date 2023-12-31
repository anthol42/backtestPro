from abc import ABC
from enum import Enum

class TradeType(Enum):
    BuyLong = 'BuyLong'
    SellLong = 'SellLong'
    SellShort = 'SellShort'
    BuyShort = 'BuyShort'

class Trade(ABC):
    def __init__(self, security: str, security_price: float, amount: int, transaction_id: str, trade_type: TradeType):
        self.security = security
        self.security_price = security_price
        self.amount = amount
        self.transaction_id = transaction_id
        self.trade_type = trade_type

    def getPrice(self) -> float:
        return self.security_price * self.amount

    def __str__(self):
        return f'[{self.trade_type}] -- {self.security}: {self.amount} x {self.security_price}'



class BuyLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, transaction_id: str):
        super().__init__(security, security_price, amount, transaction_id, trade_type=TradeType.BuyLong)
class SellLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, transaction_id: str):
        super().__init__(security, security_price, amount, transaction_id, trade_type=TradeType.SellLong)
class SellShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, transaction_id: str):
        super().__init__(security, security_price, amount, transaction_id, trade_type=TradeType.SellShort)
class BuyShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, transaction_id: str):
        super().__init__(security, security_price, amount, transaction_id, trade_type=TradeType.BuyShort)
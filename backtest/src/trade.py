from abc import ABC
from enum import Enum
from datetime import datetime
from typing import Tuple

class TradeType(Enum):
    BuyLong = 'BuyLong'
    SellLong = 'SellLong'
    SellShort = 'SellShort'
    BuyShort = 'BuyShort'


class TradeOrder(ABC):
    def __init__(self, security: str, security_price_limit: Tuple[float, float], amount: int, amount_borrowed: int,
                 trade_type: TradeType, expiry: datetime):
        self.security = security
        self.security_price_limit = security_price_limit
        self.amount = amount
        self.amount_borrowed = amount_borrowed
        self.trade_type = trade_type
        self.expiry = expiry
        self.margin_trade = amount_borrowed > 0

    def __str__(self):
        if self.margin_trade:
            return f'ORDER: [{self.trade_type} - MARGIN] -- {self.security}: {self.amount + self.amount_borrowed}'
        else:
            return f'ORDER: [{self.trade_type}] -- {self.security}: {self.amount + self.amount_borrowed}'

    def convertToTrade(self, security_price: float, timestamp: datetime, transaction_id: str):
        return Trade(self.security, security_price, self.amount, self.amount_borrowed, transaction_id,
                     timestamp, self.trade_type)

class BuyLongOrder(TradeOrder):
    def __init__(self, security: str, security_price_limit: Tuple[float, float], amount: int, amount_borrowed: int,
                  expiry: datetime):
        super().__init__(security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.BuyLong,
                         expiry=expiry)
class SellLongOrder(TradeOrder):
    def __init__(self, security: str, security_price_limit: Tuple[float, float], amount: int, amount_borrowed: int,
                  expiry: datetime):
        super().__init__(security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.SellLong,
                         expiry=expiry)
class SellShortOrder(TradeOrder):
    def __init__(self, security: str, security_price_limit: Tuple[float, float], amount: int, amount_borrowed: int,
                  expiry: datetime):
        super().__init__(security, security_price_limit, amount,  amount_borrowed, trade_type=TradeType.SellShort,
                         expiry=expiry)
class BuyShortOrder(TradeOrder):
    def __init__(self, security: str, security_price_limit: Tuple[float, float], amount: int, amount_borrowed: int,
                  expiry: datetime):
        super().__init__(security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.BuyShort,
                         expiry=expiry)

class Trade(ABC):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str, timestamp: datetime, trade_type: TradeType):
        self.security = security
        self.security_price = security_price
        self.amount = amount
        self.amount_borrowed = amount_borrowed
        self.transaction_id = transaction_id
        self.trade_type = trade_type
        self.margin_trade = amount_borrowed > 0
        self.timestamp = timestamp

    def getPrice(self) -> float:
        return self.security_price * self.amount

    def __str__(self):
        if self.margin_trade:
            return f'TRADE: [{self.trade_type} - MARGIN] -- {self.security}: {self.amount + self.amount_borrowed} x {self.security_price} || {self.timestamp}'
        else:
            return f'TRADE: [{self.trade_type}] -- {self.security}: {self.amount + self.amount_borrowed} x {self.security_price} || {self.timestamp}'



class BuyLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str, timestamp: datetime):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp, trade_type=TradeType.BuyLong)
class SellLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str, timestamp: datetime):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp, trade_type=TradeType.SellLong)
class SellShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str, timestamp: datetime):
        super().__init__(security, security_price, amount,  amount_borrowed, transaction_id, timestamp, trade_type=TradeType.SellShort)
class BuyShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str, timestamp: datetime):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp, trade_type=TradeType.BuyShort)
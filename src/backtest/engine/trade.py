# Copyright (C) 2024 Anthony Lavertu
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from abc import ABC
from enum import Enum
from datetime import datetime
from typing import Tuple, Optional

class TradeType(Enum):
    BuyLong = 'BuyLong'
    SellLong = 'SellLong'
    SellShort = 'SellShort'
    BuyShort = 'BuyShort'

    @classmethod
    def available(cls):
        return [e.value for e in cls]


class TradeOrder(ABC):
    def __init__(self, timestamp: datetime, security: str, security_price_limit: Tuple[Optional[float], Optional[float]], amount: int, amount_borrowed: int,
                 trade_type: TradeType, expiry: Optional[datetime]):
        self.timestamp = timestamp
        self.security = security
        self.security_price_limit = security_price_limit
        if amount < 0:
            raise ValueError("The amount of securities must be positive")
        if amount_borrowed < 0:
            raise ValueError("The amount of borrowed securities must be positive")
        if trade_type == TradeType.SellShort or trade_type == TradeType.BuyShort:
            if amount > 0:
                raise ValueError("Short selling orders must have only amount_borrowed since the securities are borrowed")
            if amount_borrowed <= 0:
                raise ValueError("Short selling orders must have a positive amount_borrowed")
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
                     timestamp, self.trade_type, self)

    def export(self) -> dict:
        """
        This method export the trade order object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": f"TradeOrder.{self.trade_type.value}",
            "timestamp": str(self.timestamp),
            "security": self.security,
            "security_price_limit": self.security_price_limit,
            "amount": float(self.amount),
            "amount_borrowed": float(self.amount_borrowed),
            "trade_type": self.trade_type.value,
            "expiry": str(self.expiry),
            "margin_trade": self.margin_trade,
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method loads a trade order object from a dictionary.
        :param data: The dictionary containing the object state
        :return: The object
        """
        trade_type = TradeType(data["trade_type"])
        if trade_type == TradeType.BuyLong:
            self = BuyLongOrder(datetime.fromisoformat(data["timestamp"]), data["security"], tuple(data["security_price_limit"]), data["amount"], data["amount_borrowed"],
                       datetime.fromisoformat(data["expiry"]) if data["expiry"] != "None" else None)
        elif trade_type == TradeType.SellLong:
            self = SellLongOrder(datetime.fromisoformat(data["timestamp"]), data["security"], tuple(data["security_price_limit"]), data["amount"], data["amount_borrowed"],
                       datetime.fromisoformat(data["expiry"]) if data["expiry"] != "None" else None)
        elif trade_type == TradeType.SellShort:
            self = SellShortOrder(datetime.fromisoformat(data["timestamp"]), data["security"], tuple(data["security_price_limit"]), data["amount"], data["amount_borrowed"],
                       datetime.fromisoformat(data["expiry"]) if data["expiry"] != "None" else None)
        elif trade_type == TradeType.BuyShort:
            self = BuyShortOrder(datetime.fromisoformat(data["timestamp"]), data["security"], tuple(data["security_price_limit"]), data["amount"], data["amount_borrowed"],
                       datetime.fromisoformat(data["expiry"]) if data["expiry"] != "None" else None)
        else:
            self = cls(data["timestamp"], data["security"], tuple(data["security_price_limit"]), data["amount"], data["amount_borrowed"], trade_type,
                       datetime.fromisoformat(data["expiry"]) if data["expiry"] != "None" else None)
        self.margin_trade = data["margin_trade"]
        return self

    def __eq__(self, other):
        return (self.timestamp == other.timestamp and
                self.security == other.security and
                self.security_price_limit == other.security_price_limit and
                self.amount == other.amount and
                self.amount_borrowed == other.amount_borrowed and
                self.trade_type == other.trade_type and
                self.expiry == other.expiry and
                self.margin_trade == other.margin_trade)

class BuyLongOrder(TradeOrder):
    def __init__(self, timestamp: datetime, security: str, security_price_limit: Tuple[Optional[float], Optional[float]],
                 amount: int, amount_borrowed: int,
                  expiry: Optional[datetime]):
        super().__init__(timestamp, security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.BuyLong,
                         expiry=expiry)
class SellLongOrder(TradeOrder):
    def __init__(self, timestamp: datetime,  security: str, security_price_limit: Tuple[Optional[float], Optional[float]],
                 amount: int, amount_borrowed: int,
                  expiry: Optional[datetime]):
        super().__init__(timestamp, security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.SellLong,
                         expiry=expiry)
class SellShortOrder(TradeOrder):
    def __init__(self, timestamp: datetime,  security: str, security_price_limit: Tuple[Optional[float], Optional[float]],
                 amount: int, amount_borrowed: int,
                  expiry: Optional[datetime]):
        super().__init__(timestamp, security, security_price_limit, amount,  amount_borrowed, trade_type=TradeType.SellShort,
                         expiry=expiry)
class BuyShortOrder(TradeOrder):
    def __init__(self, timestamp: datetime,  security: str, security_price_limit: Tuple[Optional[float], Optional[float]],
                 amount: int, amount_borrowed: int,
                  expiry: Optional[datetime]):
        super().__init__(timestamp, security, security_price_limit, amount, amount_borrowed, trade_type=TradeType.BuyShort,
                         expiry=expiry)

class Trade(ABC):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str,
                 timestamp: datetime, trade_type: TradeType, order: TradeOrder):
        self.trade_order = order
        self.security = security
        self.security_price = security_price
        self.amount = amount
        self.amount_borrowed = amount_borrowed
        self.transaction_id = transaction_id
        self.trade_type = trade_type
        self.margin_trade = amount_borrowed > 0
        self.timestamp = timestamp

    def getCost(self) -> float:
        """
        This method calculates the upfront cost of the trade.  It ignores the cost of borrowing the securities.
        Note:
            It doesn't include transaction costs nor the cost of borrowing the securities.
        :return: The upfront cost of the trade
        """
        return self.security_price * self.amount

    def __str__(self):
        if self.margin_trade:
            return f'TRADE: [{self.trade_type} - MARGIN] -- {self.security}: {self.amount + self.amount_borrowed} x {self.security_price} || {self.timestamp}'
        else:
            return f'TRADE: [{self.trade_type}] -- {self.security}: {self.amount + self.amount_borrowed} x {self.security_price} || {self.timestamp}'

    def export(self) -> dict:
        """
        This method export the trade object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": f"Trade.{self.trade_type.value}",
            "security": self.security,
            "security_price": float(self.security_price),
            "amount": float(self.amount),
            "amount_borrowed": float(self.amount_borrowed),
            "transaction_id": self.transaction_id,
            "trade_type": self.trade_type.value,
            "margin_trade": bool(self.margin_trade),
            "timestamp": str(self.timestamp),
            "order": self.trade_order.export()
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method loads a trade object from a dictionary.
        :param data: The dictionary containing the object state
        :return: The object
        """
        trade_type = TradeType(data["trade_type"])
        if trade_type == TradeType.BuyLong:
            self = BuyLong(data["security"], data["security_price"], data["amount"], data["amount_borrowed"],
                       data["transaction_id"], datetime.fromisoformat(data["timestamp"]), TradeOrder.load(data["order"]))
        elif trade_type == TradeType.SellLong:
            self = SellLong(data["security"], data["security_price"], data["amount"], data["amount_borrowed"],
                       data["transaction_id"], datetime.fromisoformat(data["timestamp"]), TradeOrder.load(data["order"]))
        elif trade_type == TradeType.SellShort:
            self = SellShort(data["security"], data["security_price"], data["amount"], data["amount_borrowed"],
                       data["transaction_id"], datetime.fromisoformat(data["timestamp"]), TradeOrder.load(data["order"]))
        elif trade_type == TradeType.BuyShort:
            self = BuyShort(data["security"], data["security_price"], data["amount"], data["amount_borrowed"],
                       data["transaction_id"], datetime.fromisoformat(data["timestamp"]), TradeOrder.load(data["order"]))
        else:
            self = cls(data["security"], data["security_price"], data["amount"], data["amount_borrowed"], data["transaction_id"],
                       datetime.fromisoformat(data["timestamp"]), trade_type, TradeOrder.load(data["order"]))
        self.margin_trade = data["margin_trade"]
        return self

    def __eq__(self, other):
        return (self.security == other.security and
                self.security_price == other.security_price and
                self.amount == other.amount and
                self.amount_borrowed == other.amount_borrowed and
                self.transaction_id == other.transaction_id and
                self.trade_type == other.trade_type and
                self.margin_trade == other.margin_trade and
                self.timestamp == other.timestamp and
                self.trade_order == other.trade_order)



class BuyLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str,
                 timestamp: datetime, order: TradeOrder):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp,
                         trade_type=TradeType.BuyLong, order=order)
class SellLong(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str,
                 timestamp: datetime, order: TradeOrder):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp,
                         trade_type=TradeType.SellLong, order=order)
class SellShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str,
                 timestamp: datetime, order: TradeOrder):
        super().__init__(security, security_price, amount,  amount_borrowed, transaction_id, timestamp,
                         trade_type=TradeType.SellShort, order=order)
class BuyShort(Trade):
    def __init__(self, security: str, security_price: float, amount: int, amount_borrowed: int, transaction_id: str,
                 timestamp: datetime, order: TradeOrder):
        super().__init__(security, security_price, amount, amount_borrowed, transaction_id, timestamp,
                         trade_type=TradeType.BuyShort, order=order)
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
from .account import Account
from .transaction import Transaction, TransactionType
from .broker import Broker
from abc import ABC
from datetime import datetime, timedelta
from .strategy import Strategy
from enum import Enum
from typing import Union, Tuple, Optional, List

class CashControllerTimeframe(Enum):
    __doc__ = "NO DOC"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    YEAR = "YEAR"

    def __str__(self):
        return self.value

class CashControllerBase(ABC):
    """
    Children of this class are used to control the money flow in and out of the account during the backtest.
    Each method are called at the beginning of the time period (day, week, month, year) and the user can decide to
    deposit or withdraw money from the account.  For example, If I want to deposit 1000$ every month, I would do it in
    the every_month method and 1000$ would be deposited at the beginning of every month.
    To access the account, just use the self.account attribute.  However, you must not modify the account in any way.
    To add or remove money from the account, make your overriden method return the amount of money added(Positive) or
    removed(Negative) from the account.  For example, if you want to deposit 1000$ every month, you would do:
    override the every_month method and return 1000.  If you want to withdraw 1000$ every month, you would do the same
    thing but return -1000 instead.
    If you want to do your deposit/withdrawal based on information from your portfolio or your broker, use the
    attributes: self.broker.portfolio or self.broker.
    Agan, if you use the broker, Make sure that the broker is not modified in any way!

    DO NOT override the '_init' method. Instead, override the __init__ method if you need to initialize some variables.
    """
    def __init__(self):
        self.broker: Optional[Broker] = None
        self.account: Optional[Account] = None
        self.strategy: Optional[Strategy] = None
        self._total_deposited = 0    # Record net money added or removed from account
        self.transactions: List[Transaction] = []    # Record of all transactions

    def init(self, account: Account, broker: Broker, strategy: Strategy):
        """
        :param account: The account
        :param broker: The broker [Shoudl not be modified!!]
        """
        self.broker = broker
        self.account = account
        self.strategy = strategy

    def deposit(self, timestamp: datetime, timeframe: Union[str, CashControllerTimeframe]):
        if isinstance(timeframe, str):
            timeframe = CashControllerTimeframe(timeframe)
        if timeframe == CashControllerTimeframe.DAY:
            amount, comment = self.every_day(timestamp)
            self._total_deposited += amount
            self._depo(amount, timestamp, comment)
        elif timeframe == CashControllerTimeframe.WEEK:
            amount, comment = self.every_week(timestamp)
            self._total_deposited += amount
            self._depo(amount, timestamp, comment)
        elif timeframe == CashControllerTimeframe.MONTH:
            amount, comment = self.every_month(timestamp)
            self._total_deposited += amount
            self._depo(amount, timestamp, comment)
        elif timeframe == CashControllerTimeframe.YEAR:
            amount, comment = self.every_year(timestamp)
            self._total_deposited += amount
            self._depo(amount, timestamp, comment)

    def _depo(self, amount: float, dt: datetime, comment: str):
        """
        Make a deposit if amount is positive.  If amount is negative, make a withdrawal.  If the amount is 0,
        it does nothing.
        :param amount: The amount to deposit or withdraw
        :param dt: The date of the transaction
        :param comment: The comment on the transaction
        :return: None
        """
        if amount > 0:
            self.account.deposit(amount, dt, comment=comment)
            self.transactions.append(Transaction(amount, TransactionType.DEPOSIT, dt, comment=comment))
        elif amount < 0:
            self.account.withdrawal(-amount, dt, comment=comment)
            self.transactions.append(Transaction(-amount, TransactionType.WITHDRAWAL, dt, comment=comment))

    def every_day(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        """
        This method is called every day during the backtest.
        """
        return 0., None

    def every_week(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        """
        This method is called every week during the backtest.
        """
        return 0., None

    def every_month(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        """
        This method is called every month during the backtest.
        """
        return 0., None

    def every_year(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        """
        This method is called every year during the backtest.
        """
        return 0., None

    @property
    def total_deposited(self) -> float:
        return self._total_deposited


    def monthly_variation(self, timestamp: datetime):
        threshold = timestamp - timedelta(days=30)
        return self._calc_variation(threshold, timestamp)

    def quarterly_variation(self, timestamp: datetime):
        threshold = timestamp - timedelta(days=90)
        return self._calc_variation(threshold, timestamp)

    def half_yearly_variation(self, timestamp: datetime):
        threshold = timestamp - timedelta(days=180)
        return self._calc_variation(threshold, timestamp)

    def yearly_variation(self, timestamp: datetime):
        threshold = timestamp - timedelta(days=365)
        return self._calc_variation(threshold, timestamp)

    def _calc_variation(self, t1, t2):
        """
        Calculate the variation of the account between t1 and t2
        :param t1: The initial calculation date
        :param t2: The last calculation date
        :return: The absolute variation of cash for the period
        """
        return sum([x.amount if x.t == TransactionType.DEPOSIT else -x.amount for x in self.transactions if x.dt > t1 and x.dt <= t2])


class SimpleCashController(CashControllerBase):
    """
    This class is a simple cash controller that will deposit ow withdraw a fixed amount of money at the beginning of
    every period.  If a period's amount is 0 (Default), it won't deposit or withdraw funds.
    """

    def __init__(self, every_day: float = 0., every_week: float = 0., every_month: float = 0., every_year: float = 0.):
        super().__init__()
        self.every_day_amount = every_day
        self.every_week_amount = every_week
        self.every_month_amount = every_month
        self.every_year_amount = every_year

    def every_day(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        if self.every_day_amount != 0:
            return self.every_day_amount, "Daily deposit"
        else:
            return 0., None

    def every_week(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        if self.every_week_amount != 0:
            return self.every_week_amount, "Weekly deposit"
        else:
            return 0., None

    def every_month(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        if self.every_month_amount != 0:
            return self.every_month_amount, "Monthly deposit"
        else:
            return 0., None

    def every_year(self, timestamp: datetime) -> Tuple[float, Optional[str]]:
        if self.every_year_amount != 0:
            return self.every_year_amount, "Yearly deposit"
        else:
            return 0., None
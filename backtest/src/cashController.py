from .account import Account
from .broker import Broker
from abc import ABC
from datetime import datetime
from typing import Final

class CashController(ABC):
    """
    Children of this class are used to control the money flow in and out of the account during the backtest.
    Each method are called at the beginning of the time period (day, week, month, year) and the user can decide to
    deposit or withdraw money from the account.  For example, If I want to deposit 1000$ every month, I would do it in
    the every_month method and 1000$ would be deposited at the beginning of every month.
    To access the account, just use the self.account attribute.
    If you want to do your deposit/withdrawal based on information from your portfolio or your broker, use the
    attributes: self.broker.portfolio or self.broker.
    If you use the broker, Make sure that the broker is not modified in any way!

    DO NOT override the '_init' method. Instead, override the __init__ method if you need to initialize some variables.
    """

    def _init(self, account: Account, broker: Broker):
        """
        :param account: The account
        :param broker: The broker [Shoudl not be modified!!]
        """
        self.broker: Final[Broker] = broker
        self.account = account

    def every_day(self, timestamp: datetime):
        """
        This method is called every day during the backtest.
        """
        pass

    def every_week(self, timestamp: datetime):
        """
        This method is called every week during the backtest.
        """
        pass

    def every_month(self, timestamp: datetime):
        """
        This method is called every month during the backtest.
        """
        pass

    def every_year(self, timestamp: datetime):
        """
        This method is called every year during the backtest.
        """
        pass

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import timedelta
from .account import Account
from .broker import Broker
from .record import Record, Records, RecordsBucket
from datetime import datetime
import numpy.typing as npt


class Strategy(ABC):

    def __init__(self):
        self.account: Optional[Account] = None
        self.broker: Optional[Broker] = None
        self.available_time_res: Optional[List[timedelta]] = None

    def init(self, account: Account, broker: Broker, available_time_res: List[timedelta]):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        :param account: The account object
        :param broker: The broker object
        :param available_time_res: The available time resolutions
        """
        self.account = account
        self.broker = broker
        self.available_time_res = available_time_res

    @abstractmethod
    def run(self, data: RecordsBucket, timestep: datetime):
        """
        This method is used to compute the strategy at each time step.  It is in this method that the strategy logic is
        implemented.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        raise NotImplementedError("run method not implemented")


    def __call__(self, data: RecordsBucket, timestep: datetime):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        This method is used to compute the strategy at each time step and some other computations for stats purposes.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        self.run(data, timestep)

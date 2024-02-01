from abc import ABC, abstractmethod
from typing import List, Dict
from .tsData import TSData
from .account import Account
from .broker import Broker
from .record import Record
from datetime import datetime


class Strategy(ABC):
    def __init__(self, account: Account, broker: Broker):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        :param account: The account object
        :param broker: The broker object
        """
        self.account = account
        self.broker = broker

    def init(self):
        """
        This method has no parameters and is used to initialize the strategy.
        It is called at the beginning of the simulation, before the first call to eval.
        """
        pass

    @abstractmethod
    def run(self, data: List[List[Record]], timestep: datetime):
        """
        This method is used to compute the strategy at each time step.  It is in this method that the strategy logic is
        implemented.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        raise NotImplementedError("eval method not implemented")



    def __call__(self, data: List[List[Record]], timestep: datetime):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        This method is used to compute the strategy at each time step and some other computations for stats purposes.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        self.run(data)

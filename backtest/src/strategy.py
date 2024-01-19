from abc import ABC, abstractmethod
from typing import List, Dict
from .tsData import TSData
from .account import Account
from .broker import Broker


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
    def eval(self, data: List[Dict[str, TSData]]):
        """
        This method is used to compute the strategy at each time step.  It is in this method that the strategy logic is
        implemented.
        """
        raise NotImplementedError("eval method not implemented")



    def __call__(self, data: List[Dict[str, TSData]]):
        """
        This method is used to compute the strategy at each time step and some other computations for stats purposes.
        """
        self.eval(data)
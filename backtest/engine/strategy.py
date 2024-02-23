from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import timedelta
from .account import Account
from .broker import Broker
from .record import Record
from datetime import datetime


class Strategy(ABC):

    def __init__(self):
        self.account = None
        self.broker = None

    def init(self, account: Account, broker: Broker, available_time_res: List[timedelta]):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        :param account: The account object
        :param broker: The broker object
        """
        self.account = account
        self.broker = broker
        self.available_time_res = available_time_res

    @abstractmethod
    def run(self, data: List[List[Record]], timestep: datetime):
        """
        This method is used to compute the strategy at each time step.  It is in this method that the strategy logic is
        implemented.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        raise NotImplementedError("run method not implemented")

    def indicators(self, data: List[List[Record]], timestep: datetime):
        """
        This method is used to compute the dynamic indicators at each time step.  It is strongly recommended to
        calculate indicators dynamically (even though it is slower) because it has the right price (split adjusted)
        and it is easier to use in inference mode (live trading).
        To add indicators, you can override this method and modify the data.  It is recommended to add the indicators
        to the dataframes inside each records.
        Example:
            >>> # Calculate a moving average with a 14 days period
            >>> for time_res in data:
            >>>     for record in time_res:
            >>>         record.chart['ma14'] = record.chart['Close'].rolling(window=14).mean()
            >>> return data
            >>> # In the preceding example, the data was mutated.  It doesn't matter in this method, but the data still
            >>> # must be returned.
        :param data: The data to use for the indicators
        :param timestep: The current time step
        """
        return data



    def __call__(self, data: List[List[Record]], timestep: datetime):
        """
        YOU SHOULD NOT OVERRIDE THIS METHOD
        This method is used to compute the strategy at each time step and some other computations for stats purposes.
        :param data: The data to use for the strategy
        :param timestep: The current time step
        """
        indicator_data = self.indicators(data, timestep)
        self.run(indicator_data)

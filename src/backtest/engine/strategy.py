"""
Copyright (C) 2024 Anthony Lavertu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import timedelta
from .account import Account
from .broker import Broker
from .record import Record, Records, RecordsBucket
from datetime import datetime
import numpy.typing as npt
from pathlib import PurePath
import pickle


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


    def save(self, path: PurePath):
        """
        This method is used to save the state of the strategy to a file.  If you would like to save in an other format,
        you can override this method and the load method.
        Note:
            To avoid saving things twice, we set the account, broker and available_time_res to None before saving.
            Only the other attributes will be saved.  (If any)
        :param path: The path to save the strategy (.pkl)
        """
        # Before saving, we set the account and broker to None to avoid pickling them.
        # (They should be saved separately as json)
        self.broker = None
        self.account = None
        self.available_time_res = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: PurePath):
        """
        This method is used to load the state of a strategy from a file.  If you would like to save in an other format,
        you can override this method and the save method.
        :param path: The path to load the strategy from.  (.pkl)
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

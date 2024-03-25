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
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from datetime import timedelta
from .tsData import TSData
from enum import Enum
import pandas as pd
import numpy as np

class TimeResExtender(ABC):
    # Number of new time resolution that this extender will add
    n_out: int
    out_res: List[timedelta]
    def __init__(self):
        if self.n_out is None:
            raise ValueError("n_out must be specified.")
        if self.out_res is None:
            raise ValueError("out_res must be specified.")

    def extend(self, data: List[Dict[str, TSData]], main_timestep: int = 0) -> List[Dict[str, TSData]]:
        """
        Extend the data with new time resolutions.  By default, it uses the single_extend method to extend each
        timeseries data independently.  However, it is possible to override this method and bypass the single_extend
        method to extend the data in a more complex way or to optimize the process.
        :param data: The current data statically acquired.  (Fetch from sources)
        :param main_timestep: The index of the timeseries data in data list to use as the main series.
        :return: The new data with the new time resolutions.  (Do not mutate the original data)
        """
        out = [{} for _ in range(self.n_out)]
        for ticker in data[main_timestep].keys():
            news = self.single_extend(data[main_timestep][ticker])
            for i, new in enumerate(news):
                out[i][ticker] = new
        return out

    def single_extend(self, data: TSData) -> Tuple[TSData, ...]:
        """
        Extend a single timeseries data with new time resolutions.  Override this method
        :param data: The current data statically acquired.  (Fetch from sources)
        :return: The new data with the new time resolutions.  (Do not mutate the original data)
        """
        raise NotImplementedError("This method must be implemented.")

    def __add__(self, other):
        new = deepcopy(self)
        new.n_out += other.n_out
        new.out_res += other.out_res
        new.extend = lambda *args, **kwargs: self.extend(*args, **kwargs) + other.extend(*args, **kwargs)
        return new

    def export(self):
        return {"type": "TimeResExtender", "n_out": self.n_out, "out_res": [res.total_seconds() for res in self.out_res]}

class TimePeriod(Enum):
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class BasicExtender(TimeResExtender):
    """
    This class provide an easy top implement extender for basic time resolutions such as 30 minutes, 1 hour,
    4 hours, 1 day, 1 week, and 1 month.  It assumes to have has main timestep (In the backtest object) a time
    resolution smaller than the value chose.  For example, if the main timestep is 1 day, it will not work to
    extend the data to 30 minutes or 4h.  If you have multiple static timestep that you want to extend, you can
    override the extend method to handle this case, or directly derive a new class from TimeResExtender.

    Example:
        >>> extender = BasicExtender(TimePeriod.ONE_DAY) + BasicExtender("1w")
        >>> # Given that the main timeresolution of the static data is smaller than one day.  Ex: 30 minutes
        >>> # The data will have a time resolution of 30 minutes, 1 day and 1 week.
        >>> data: List[Dict[str, TSData]] = ...
        >>> data += extender.extend(data)
    """
    def __init__(self, period: Union[TimePeriod, str]):
        self.n_out = 1
        if isinstance(period, str):
            period = TimePeriod(period)
        self.period = period
        if period == TimePeriod.THIRTY_MINUTES:
            self.out_res = [timedelta(minutes=30)]
        elif period == TimePeriod.ONE_HOUR:
            self.out_res = [timedelta(hours=1)]
        elif period == TimePeriod.FOUR_HOURS:
            self.out_res = [timedelta(hours=4)]
        elif period == TimePeriod.ONE_DAY:
            self.out_res = [timedelta(days=1)]
        elif period == TimePeriod.ONE_WEEK:
            self.out_res = [timedelta(weeks=1)]
        elif period == TimePeriod.ONE_MONTH:
            self.out_res = [timedelta(days=30)]
        else:
            raise ValueError("Invalid period.")
        super().__init__()

    def single_extend(self, data: TSData) -> Tuple[TSData]:
        if data.time_res.total_seconds() > self.out_res[0].total_seconds():
            raise RuntimeError("The main timestep must be smaller than the extended timestep.")
        pd.set_option('mode.chained_assignment', None)
        out: TSData
        if self.period == TimePeriod.THIRTY_MINUTES:
            out = self.thirty_minutes(data)
        elif self.period == TimePeriod.ONE_HOUR:
            out = self.one_hour(data)
        elif self.period == TimePeriod.FOUR_HOURS:
            out = self.four_hours(data)
        elif self.period == TimePeriod.ONE_DAY:
            out = self.one_day(data)
        elif self.period == TimePeriod.ONE_WEEK:
            out = self.one_week(data)
        elif self.period == TimePeriod.ONE_MONTH:
            out = self.one_month(data)
        else:
            pd.set_option('mode.chained_assignment', 'warn')
            raise RuntimeError("Invalid period.")
        return out,

    def thirty_minutes(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "30T")
        out.time_res = timedelta(minutes=30)
        return out

    def one_hour(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "h")
        out.time_res = timedelta(hours=1)
        return out

    def four_hours(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "4h")
        out.time_res = timedelta(hours=4)
        return out

    def one_day(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "D")
        out.time_res = timedelta(days=1)
        return out

    def one_week(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "W")
        out.time_res = timedelta(weeks=1)
        return out

    def one_month(self, data: TSData) -> TSData:
        out = deepcopy(data)
        out.data = self._resample(out, "30D")
        out.time_res = timedelta(days=30)
        return out

    @staticmethod
    def _resample(data: TSData, sample_s: str) -> pd.DataFrame:
        """
        This method will mutate the data.data attribute.  Make sure to make a deep copy before using this method.
        :param data: The data to resample
        :param sample_s: The sample indicator
        :return: The resampled dataset
        """
        data = data.data
        open_resampled = data["Open"].resample(sample_s, closed='right', label='left').ohlc()['open']
        high_resampled = data["High"].resample(sample_s, closed='right', label='left').ohlc()["high"]
        low_resampled = data["Low"].resample(sample_s, closed='right', label='left').ohlc()["low"]
        close_resampled = data["Close"].resample(sample_s, closed='right', label='left').ohlc()["close"]
        ohlc_resampled = pd.concat([open_resampled, high_resampled, low_resampled, close_resampled], axis=1)
        ohlc_resampled.columns = ["Open", "High", "Low", "Close"]
        # resample other columns such as volume and dividends
        if "Stock Splits" in data.columns:
            data.loc[data["Stock Splits"] == 0, "Stock Splits"] = 1
            stock_splits_resampled = data["Stock Splits"].resample(sample_s, closed='right', label='left').prod()
            stock_splits_resampled.loc[stock_splits_resampled == 1] = 0
            other_resampled = data.drop(columns=["Open", "High", "Low", "Close", "Stock Splits"]).\
                               resample(sample_s, closed='right', label='left').sum()
            other_resampled["Stock Splits"] = stock_splits_resampled
        else:
            other_resampled = data.drop(columns=["Open", "High", "Low", "Close"]).\
                resample(sample_s, closed='right', label='left').sum()
        # combine OHLC and other columns into one DataFrame
        out = pd.concat([ohlc_resampled, other_resampled], axis=1)
        # Drop nan rows
        out = out.iloc[~np.isnan(close_resampled.values)]
        return out

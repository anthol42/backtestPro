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
import pandas as pd
from typing import Optional, Union, List, Dict, Iterable
from .tsData import DividendFrequency
from datetime import timedelta
import numpy as np
import numpy.typing as npt

class Record:
    """
    This class contains information about a stock and is pass to the strategy.
    """
    def __init__(self, chart: Optional[pd.DataFrame], ticker: str, time_res: int, marginable: bool, shortable: bool,
                 div_freq: DividendFrequency,
                 short_rate: float,
                 next_tick: Optional[pd.Series] = None):
        """

        :param chart: The prepared chart of the stock
        :param ticker: The ticker of the stock
        :param time_res: The idx in the list of resolution of the backtest object of the time resolution of the record.
        :param marginable: Whether the stock is marginable
        :param shortable: Whether the stock is shortable
        :param div_freq: The dividend frequency of the stock
        :param short_rate: The short rate of the stock
        :param next_tick: The next tick of the stock
        """
        self.chart = chart
        # The strategy should not use this attribute.  It is only for the broker
        self._next_tick: pd.Series = next_tick
        self.short_rate = short_rate
        self.ticker = ticker
        self.time_res = time_res
        self.marginable = marginable
        self.shortable = shortable
        self.div_freq = div_freq
        self.has_dividends = div_freq != DividendFrequency.NO_DIVIDENDS

    @property
    def next_tick(self):
        """
        This method is used to get the next tick of the stock.
        :return: The next tick of the stock
        """
        return self._next_tick

    def __eq__(self, other):
        return self.chart.equals(other.chart) and self.ticker == other.ticker and self.time_res == other.time_res and \
                self.marginable == other.marginable and self.shortable == other.shortable and self.div_freq == other.div_freq and \
                self.short_rate == other.short_rate and self.next_tick.equals(other.next_tick)


class Records:
    """
    Class containing multiple records for a given time resolution.
    """
    def __init__(self, records: Union[List[Record], Dict[str, Record]], time_res: timedelta, time_res_idx: int,
                 window: int):
        """
        :param records: The records.  If a list, it will be converted to a dictionary internally.  If a dictionary, the
                                        keys will be the tickers of the securities.
        :param time_res: The time resolution
        :param time_res_idx: The index of the time resolution in the available_time_res attribute of the backtest object
        :param window: The window size (lookback period) of the records.  Used when exporting to numpy.
        """
        if isinstance(records, list):
            records = {record.ticker: record for record in records}
        self.records = records
        self.tickers = list(records.keys())
        self.features = None
        self.update_features()
        self.time_res = time_res
        self.time_res_idx = time_res_idx
        self.window = window

    def __getitem__(self, item: Union[str, int]) -> Record:
        if isinstance(item, int):
            return self.records[self.tickers[item]]
        else:
            return self.records[item]

    def __setitem__(self, key: Union[str, int], value: Record):
        if isinstance(key, int):
            self.records[self.tickers[key]] = value
        else:
            self.records[key] = value

    def __iter__(self):
        return iter(self.records.items())

    def numpy(self) -> npt.NDArray[np.float64]:
        out = []
        for ticker in self.tickers:
            chart = self.records[ticker].chart.to_numpy()
            if len(chart) < self.window:
                chart = np.concatenate((np.full((self.window - len(chart), chart.shape[1]), np.nan), chart), axis=0)
            out.append(chart)

        return np.array(out)

    def to_list(self) -> List[Record]:
        return [self.records[ticker] for ticker in self.tickers]

    def update_features(self):
        """
        This method will update the features attribute to the new features set based on the columns of the charts.
        Assuming all the charts have the same features.
        :return: None
        """
        self.features = self.records[self.tickers[0]].chart.columns



class RecordsBucket:
    """
    Class containing multiple records for all time resolutions.
    """
    def __init__(self, records: Union[List[Records], List[Iterable[Record]]], available_time_res: List[timedelta],
                 main_timestep: int, window: Optional[int] = None):
        """
        :param records: The records
        :param available_time_res: The available time resolutions
        """
        if isinstance(records[0], Records):
            self.records = {record.time_res: record for record in records}
        else:
            if window is None:
                raise ValueError("The window parameter must be specified when records is a list of iterable of records")
            self.records = {available_time_res[i]: Records(list(records[i]), available_time_res[i], i, window) for i in range(len(records))}
        self.available_time_res = available_time_res
        self.main_timestep = main_timestep

    def __getitem__(self, item: Union[timedelta, int]) -> Records:
        """
        Get a set of Records by time resolution
        :param item: The time resolution
        :return:
        """
        if isinstance(item, int):
            return self.records[self.available_time_res[item]]
        else:
            return self.records[item]

    def __setitem__(self, key: Union[timedelta, int], value: Records):
        """
        Set a set of Records by time resolution
        :param key: The time resolution
        :param value: The Records
        """
        if isinstance(key, int):
            self.records[self.available_time_res[key]] = value
        else:
            self.records[key] = value

    def __iter__(self):
        return iter(self.records.items())

    @property
    def main(self) -> Records:
        """
        Get the main timestep set of records
        :return: The main timestep set of records
        """
        return self.records[self.available_time_res[self.main_timestep]]

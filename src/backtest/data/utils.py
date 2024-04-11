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
from .pipes import Fetch, Process, Collate, Cache, PipeOutput, RevalidateAction, CacheObject, DataPipeType, DataPipe
from datetime import datetime, timedelta
import json
import pickle
from typing import Optional, Tuple, Any, Callable, Iterable, Dict, Union, List
from . import json_extension as je
import os
import yfinance as yf
from tqdm import tqdm
from ..engine import TSData
import time

class JSONCacheObject(CacheObject):
    def store(self):
        if not os.path.exists(".cache"):
            os.mkdir(".cache")
        with open(f".cache/{self.pipe_id}.json", "w") as file:
            out = {
                "value": self.value,
                "pipe_id": self.pipe_id,
                "write_time": self.write_time,
                "next_revalidate": self.next_revalidate,
                "max_request": self.max_request,
                "current_n_request": self.current_n_requests,
                "hash": self.pipe_hash
            }
            json.dump(out, file, cls=je.JSONEncoder)

    @classmethod
    def load(cls, pipe_id: int) -> 'JSONCacheObject':
        with open(f".cache/{pipe_id}.json", "r") as file:
            data = json.load(file, cls=je.JSONDecoder)
            new_cache = cls(data["value"], data["pipe_id"], data["hash"], data["next_revalidate"],
                       data["max_request"], data["current_n_request"])
            new_cache.write_time = data["write_time"]
        return new_cache

class JSONCache(Cache):
    """
    This pipe will cache the data as a JSON file instead of a pickle file.  This class uses an extended JSON encoder
    and decoder to encode most datatypes.  However, the encoded version might be suboptimal.  To make objects
    JSON serializable, you can add a __tojson__ method that returns a JSONable dictionary to your object.  To make it
    loadable, you can add a __fromjson__ class method that takes a JSONable dictionary and returns an instance of your
    object.
    """
    def __init__(self, *,
                 revalidate: Optional[datetime] = None,
                 timeout: Optional[timedelta] = None,
                 max_requests: Optional[int] = None,
                 store: bool = True,
                 revalidate_cb: Callable[[datetime, datetime, Tuple[Any, ...], PipeOutput, dict[str, Any]],
                    RevalidateAction] = None,
                 custom_types: Optional[dict[str, type]] = None):
        """
        :param revalidate: The datetime to revalidate the cache
        :param timeout: The timedelta at which the cache will expire and be revalidated periodically
        :param max_requests: The maximum number of request before revaidating the cache
        :param store: Whether to store on disk or only in memory
        :param revalidate_cb: A revalidate callback function to implement a custom revalidation mechanism
        :param custom_types: The custom types to add to the JSONEncoder and JSONDecoder to make them JSON serializable
        and loadable.  Every type that implemented the __tojson__ and __fromjson__ methods should be registered here.
        <class_name: class>
        """
        super().__init__(revalidate=revalidate, timeout=timeout, max_requests=max_requests,
                         store=store, revalidate_cb=revalidate_cb)
        self._custom_types = custom_types if custom_types is not None else {}
        self.name = "JSONCache"

    def __call__(self, *args, **kwargs):
        """
        This object is not implemented for JSONCache, use Cache pipe instead.  This method is
        not meant to be used as a decorator nor to be called directly.
        :param args:
        :param kwargs:
        :raise NotImplementedError:
        """
        raise NotImplementedError("This method is not implemented for JSONCache, use Cache pipe instead")

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> None:
        """
        This method is called to cache the data.  This emthod will cache the data to the json format, and save it
        to the disk.  It will also save as pickle the automatically detected types.
        :param frm: Start Datetime
        :param to: End Datetime
        :param args: Any arguments passed to the pipeline
        :param po: The previous PipeOutput
        :param kwargs: Any keyword arguments passed to the pipeline
        :return: None
        """
        self._cache = JSONCacheObject(po.value, self._pipe_id, self.hash(), self._revalidate, self._max_request,
                                      self._n_requests)
        if self.store:
            je.add_types(**self._custom_types)
            self._cache.store()
            with open(".cache/detected_types.pkl", "wb") as file:
                pickle.dump(je.get_detected_types(), file)
            je.remove_types(*self._custom_types.keys())

    def load(self) -> Optional[JSONCacheObject]:
        """
        This method is called to load the cache.  This method will load the cache from the disk, and return the
        CacheObject.  If the cache is not found, it will return None.
        :return: JSONCacheObject if cache is found and None otherwise
        """
        if os.path.exists(f".cache/{self._pipe_id}.json"):
            je.add_types(**self._custom_types)
            with open(".cache/detected_types.pkl", "rb") as file:
                je.add_detected_types(pickle.load(file))
            out = JSONCacheObject.load(self._pipe_id)
            je.remove_types(*self._custom_types.keys())
            return out
        else:
            return None


class FetchCharts(DataPipe):
    """
    This pipe will fetch the charts from Yahoo Finance.  It will fetch the charts for the specified tickers and
    the specified interval.  The interval can be any interval supported by yfinance.  The data will be returned as a
    dictionary of DataFrames where the keys are the tickers and the values are the DataFrames.  The columns of the
    returned dataframes are the following: Open, High, Low, Close, Volume, Dividends, Stock Splits.  The index is named
    "Date" for intervals of 1d and longer and "Datetime" for intervals shorter than 1d.  The index is a datetime index.
    If the returned chart is empty, it will be returned as None.  Consider removing the None  charts in further
    preprocessing steps.
    Warning:
        The returned index contains the timezone information that might be inconsistent.  Consider removing them.

    IN: Optional[List[str]]: The list of tickers to fetch the charts for.  If provided, it will override the ticker
                                list passed during initialization.
    OUT: dict[str, Optional[pd.DataFrame]] where the values are the charts and the keys are the tickers
    """
    def __init__(self, tickers = None, interval: str = "1d", progress: bool = False, throttle: float = 0.,
                 *args, **kwargs):
        super().__init__(T=DataPipeType.FETCH)
        self.name = "FetchCharts"
        self.tickers = tickers
        self.interval = interval
        self.progress = progress
        self.throttle = throttle
        self.args = args
        self.kwargs = kwargs

    def fetch(self, frm: datetime, to: datetime, *args, po, **kwargs) -> PipeOutput:
        if po is not None and po.value is not None:
            tickers = po.value
        else:
            tickers = self.tickers
        charts: Dict[str, Optional[pd.DataFrame]] = {}
        if self.progress:
            for ticker in tqdm(tickers, desc="Fetching Charts"):
                charts[ticker] = yf.Ticker(ticker).history(start=frm, end=to, interval=self.interval,
                                                           *self.args, **self.kwargs)
                if charts[ticker].empty:
                    charts[ticker] = None
                time.sleep(self.throttle)
        else:
            for ticker in tickers:
                charts[ticker] = yf.Ticker(ticker).history(start=frm, end=to, interval=self.interval,
                                                           *self.args, **self.kwargs)
                if charts[ticker].empty:
                    charts[ticker] = None
                time.sleep(self.throttle)
        return PipeOutput(charts, self)

@Process
def FilterNoneCharts(frm: datetime, to: datetime, *args, po: PipeOutput[Dict[str, Optional[pd.DataFrame]]],
                     **kwargs) -> Dict[str, pd.DataFrame]:
    """
    This pipe will filter out the tickers that doesn't have a chart.  (Chart is None)
    IN: dict[str, Optional[pd.DataFrame]] where the values are the charts and the keys are the tickers
    OUT: dict[str, pd.DataFrame] where the values are the charts and the keys are the tickers
    :param frm: From datetime
    :param to:  to datetime
    :param args: Args passed to the pipe
    :param po: Previous Pipe output
    :param kwargs: The keyword arguments passed to the pipe
    :return:
    """
    return {k: v for k, v in po.value.items() if v is not None}


@Process
def ToTSData(frm: datetime, to: datetime, *args, po: PipeOutput[Dict[str, pd.DataFrame]],
             **kwargs) -> List[Dict[str, TSData]]:
    """
    This pipe will convert the charts into a time series object (TSData).
    IN: dict[str, pd.DataFrame] where the values are the charts and the keys are the tickers
    OUT: list[dict[str, TSData]] where the values are the TSData and the keys are the tickers (len of list = 1)
    :param frm: From datetime
    :param to:  to datetime
    :param args: Args passed to the pipe
    :param po: Previous Pipe output
    :param kwargs: The keyword arguments passed to the pipe
    :return: pd.DataFrame
    """
    return [{ticker: TSData(chart, name=f"Chart-{ticker}") for ticker, chart in po.value.items()}]


@Process
def CausalImpute(frm: datetime, to: datetime, *args, po: PipeOutput[Dict[str, pd.DataFrame]],
                 **kwargs) -> Dict[str, pd.DataFrame]:
    """
    This pipe will impute the missing values in the time series data using the causal imputation method. (It will copy
    the last time step value to the missing values)
    Example:
        a = [1, 2, 3, nan, 5, 6, nan, 8, 9]
        causal_impute(a) -> [1, 2, 3, 3, 5, 6, 6, 8, 9]
    IN: dict[str, pd.Dataframe] where the values are the charts as Dataframe and the keys are the tickers
    OUT: dict[str, TSData] where the values are the TSData and the keys are the tickers
    :param frm: From datetime
    :param to:  to datetime
    :param args: Args passed to the pipe
    :param po: Previous Pipe output
    :param kwargs: The keyword arguments passed to the pipe
    :return: pd.DataFrame
    """
    return {ticker: chart.ffill() for ticker, chart in po.value.items()}


@Process
def PadNan(frm: datetime, to: datetime, *args, po: PipeOutput[Dict[str, Optional[pd.DataFrame]]], **kwargs):
    """
    This pipe will pad the charts (dataframes) with NaNs.  It will reindex them with the longest index found in the
    data.  The index will be the same for all the charts.
    :param frm: From datetime
    :param to:  to datetime
    :param args: Args passed to the pipe
    :param po: Previous Pipe output. Must be a dictionary of charts (ticker, pd.DataFrame)
    :param kwargs: The keyword arguments passed to the pipe
    :return: pd.DataFrame
    """
    data = po.value
    index = None
    out = {}
    # Get the longest index
    for ticker, chart in data.items():
        if index is None:
            index = chart.index
            continue

        if len(chart.index) > len(index):
            index = chart.index

    # Pad the dataframes with NaNs
    for ticker, chart in data.items():
        out[ticker] = chart.reindex(index)

    return out
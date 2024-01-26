from datetime import timedelta
import numpy as np
import pandas as pd
from enum import Enum

class DividendFrequency(Enum):
    NO_DIVIDENDS = 0
    MONTHLY = 1
    QUARTERLY = 2
    BIANNUALLY = 3
    YEARLY = 4

    @classmethod
    def from_delta(cls, days: float):
        days = cls.nearest(days, np.array([30, 90, 180, 365], dtype=np.int32))
        if days == 30:
            return cls.MONTHLY
        elif days == 90:
            return cls.QUARTERLY
        elif days == 180:
            return cls.BIANNUALLY
        else:
            return cls.YEARLY

    @classmethod
    def nearest(cls, value: float, available_values: np.ndarray) -> int:
        """
        Get the nearest value in available_values.  (Kinda like a custom round())
        :param value: The value to approximate
        :param available_values: The available values to approximate to
        :return: The approximation (nearest value)
        """
        return available_values[np.abs(available_values - value).argmin()]

class TSData:
    """
    This class is is used to store timeseries data and make them usable for the Backtest object
    """

    def __init__(self, data: pd.DataFrame, *, name: str = "Default", time_res: timedelta = None, div_freq: DividendFrequency = None):
        """
        :param data: Data is a pandas OHLCV+ dataframe meaning it should have at least the following
                        columns: Open, High, Low, Close, Volume, Stock Splits
        :param name: Name of timeseries data
        :param time_res: The resolution of the data
        :param div_freq: The frequency that the security is paying dividends.  If None, then the frequency is approximated.
        """
        self.data = data
        self.time_res: timedelta = data.index.diff().min() if time_res is None else time_res
        self.name = name

        # Find start and end of data since data is padded using nan
        start, end = self._find_start_end()
        self._start = self.data.index[start]
        self._end = self.data.index[end]

        if div_freq is None:
            if "Dividends" not in data.columns:
                self.div_freq = DividendFrequency.NO_DIVIDENDS
            else:
                idx_div = data.index[(data["Dividends"] > 0)]
                if len(idx_div) == 0:
                    self.div_freq = DividendFrequency.NO_DIVIDENDS
                else:
                    self.div_freq = DividendFrequency.from_delta(idx_div.diff().days.to_numpy()[1:].mean())
        else:
            self.div_freq = div_freq

    def _find_start_end(self):
        array = self.data.to_numpy()
        start_idx = np.argmin(np.isnan(array).any(axis=1))
        end_idx = np.argmax(np.isnan(array[start_idx:]).any(axis=1))
        return start_idx, end_idx

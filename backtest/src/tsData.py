from datetime import timedelta
import numpy as np
import pandas as pd


class TSData:
    """
    This class is is used to store timeseries data and make them usable for the Backtest object
    """

    def __init__(self, data: pd.DataFrame, *, name: str = "Default", time_res: timedelta = None):
        self.data = data
        self.time_res: timedelta = data.index.diff().min() if time_res is None else time_res
        self.name = name

        # Find start and end of data since data is padded using nan
        start, end = self._find_start_end()
        self._start = self.data.index[start]
        self._end = self.data.index[end]

    def _find_start_end(self):
        array = self.data.to_numpy()
        start_idx = np.argmin(np.isnan(array).any(axis=1))
        end_idx = np.argmax(np.isnan(array[start_idx:]).any(axis=1))
        return start_idx, end_idx


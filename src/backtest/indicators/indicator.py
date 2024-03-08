import pandas as pd
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import List, Tuple, Union, Optional, Any, Callable


class Indicator:
    """
    This class is a decorator designed to make an Indicator out of a function.
    Example:
        >>> @Indicator(out_feat=["SMA"], period=int)
        ... def SMA(data: np.ndarray, index: List[str], features: List[str], period: int = 10) -> np.ndarray:
        ...     out = np.zeros(len(data), dtype=np.float32)
        ...     for i in range(len(data) - period + 1):
        ...         out[i] = data[i: i+period, 0].sum() / period
        ...     return out[:, np.newaxis]
        >>> # Usage
        >>> series = pd.DataFrame(np.random.rand(1000), index=pd.date_range(start="2021-01-01", periods=1000, freq="min"), columns=["Close"])
        >>> indicator_10d = SMA()
        >>> indicator_20d = SMA(period=20)
        >>> out_10d = indicator_10d.get(series)
        >>> out_20d = indicator_20d.get(series)

    The indicator function receives the data as a 2D numpy array, the index as a list of datetimes (or string if jit
    compiled), the features as a list of strings, and any parameters set by the user.  The function must return a 2D
    numpy array with the same length as the input data.
    """
    def __init__(self, out_feat: List[str], numba: bool = False, name: Optional[str] = None, **expected_params: type):
        """
        :param out_feat: The features name that this indicator will return.  (Will be the column name in resulting df)
        :param numba: Whether the function uses numba JIT compilation or not.  If True, the index passed to the
                        indicator logic will be a list of strings, otherwise a list of datetime objects.
        :param name: The name of the indicator.  If not set, the name of the function will be used.
        :param expected_params: The expected parameters of the indicator.  Used to validate the parameters passed to the
                                indicator.
        """
        self.out = out_feat
        self.use_numba = numba
        self._cb: Optional[Callable[[npt.NDArray[np.float32], Union[List[datetime], List[str]],
                                     List[str], ...],  npt.NDArray[np.float32]]] = None
        self.name = name
        self.params = {}
        self.expected_params = expected_params


    def __call__(self, *args, **kwargs) -> 'Indicator':
        """
        This method is called when the decorator is used to decorate a function and when the object is 'initialized'.
        There is a single argument when used as a decorator, the function to be decorated, and multiple kwargs when
        initialized.
        :param args: a single argument when used as a decorator, the function to be decorated.
        :param kwargs: The indicator parameters
        :return:
        """
        if len(args) > 0 and callable(args[0]):
            self.set_callback(args[0])
        else:
            self.set_params(**kwargs)
        return self

    def set_callback(self, cb: Callable[[npt.NDArray[np.float32], Union[List[datetime], List[str]],
                                   List[str], ...],  npt.NDArray[np.float32]]) -> None:
        """
        Set the callback method to be used when the indicator is called.
        :param cb: The indicator logic
        :return: None
        """
        if self.name is None:
            self.name = cb.__name__
        self._cb = cb

    def set_params(self, **params: type) -> None:
        """
        Set the parameters of the indicator.
        :param params: The parameters set by the user
        :return: None
        """
        for param in params:
            if param not in self.expected_params:
                raise ValueError(f"Invalid parameter: {param}.  Expect: {self.expected_params.keys()}")
            if not isinstance(params[param], self.expected_params[param]):
                raise ValueError(f"Invalid type for parameter {param}.  Expect: {self.expected_params[param]}")
        self.params = params


    def get(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the indicator logic on the data and returns the result as a DataFrame.
        :param data: The input dataframe with a datetime index.  (Usually the chart)
        :return: The indicator results as a dataframe
        """
        np_data = data.to_numpy()
        if self.use_numba:
            index = [t.isoformat() for t in data.index.tolist()]
        else:
            index = data.index.tolist()
        features = data.columns.tolist()
        if self._cb is None:
            out = self.run(np_data, index, features, **self.params)
        else:
            out = self._cb(np_data, index, features, **self.params)
        return pd.DataFrame(out, index=index, columns=self.out)

    @staticmethod
    def run(data: npt.NDArray[np.float32], index: Union[List[datetime], List[str]], features: List[str],
            **params) -> npt.NDArray[np.float32]:
        """
        This method can be overridden to implement the indicator logic.  This method is called when the callback method
        is not set.
        :param data: The input data as a 2d numpy array.
        :param index: The index of the input data.  (datetime if not compiled with numba or string if compiled
        :param features: The features of the input data
        :param params: Any parameters set by the user
        :return: A 2D array with the indicator results.  Must be the same length as the input data.
        """
        raise NotImplementedError("This method must be implemented in a subclass, or you must set the callback.")


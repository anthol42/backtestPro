import pandas as pd
import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import List, Tuple, Union, Optional, Any, Callable
from copy import deepcopy


class Indicator:
    """
    This class is a decorator designed to make an Indicator out of a function.
    Example:
        >>> @Indicator(out_feat=["SMA"], period=int)
        ... def SMA(data: np.ndarray, index: List[str], features: List[str],
        ...     previous_data: Optional[npt.NDArray[np.float32]], period: int = 10) -> np.ndarray:
        ...
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
        self._out = out_feat
        self._id = 0
        self.use_numba = numba
        self._cb: Optional[Callable[[npt.NDArray[np.float32], Union[List[datetime], List[str]],
                                     List[str], Optional[npt.NDArray[np.float32]], ...],  npt.NDArray[np.float32]]] = None
        self._name = name
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
            return self
        else:
            new = deepcopy(self)
            new.set_params(**kwargs)
            return new

    def set_callback(self, cb: Callable[[npt.NDArray[np.float32], Union[List[datetime], List[str]],
                                   List[str], Optional[npt.NDArray[np.float32]], ...],
                                    npt.NDArray[np.float32]]) -> None:
        """
        Set the callback method to be used when the indicator is called.
        :param cb: The indicator logic
        :return: None
        """
        if self._name is None:
            self._name = cb.__name__
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

    def get(self, data: pd.DataFrame, previous_values: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Run the indicator logic on the data and returns the result as a DataFrame.
        :param data: The input dataframe with a datetime index.  (Usually the chart)
        :param previous_values: The previous values of the indicator.  (Used when streaming the indicator)
        :return: The indicator results as a dataframe
        """
        np_data = data.to_numpy()
        if self.use_numba:
            index = [t.isoformat() for t in data.index.tolist()]
        else:
            index = data.index.tolist()
        features = data.columns.tolist()
        if previous_values is not None:
            previous_values = previous_values[self.out].to_numpy()
        else:
            previous_values = None
        if self._cb is None:
            out = self.run(np_data, index, features, previous_values, **self.params)
        else:
            out = self._cb(np_data, index, features, previous_values, **self.params)

        return pd.DataFrame(out, index=index, columns=self.out)

    @property
    def out(self) -> List[str]:
        """
        :return: The output features of the indicator
        """
        if self._id != 0:
            return [f"{feat}_{self._id}" for feat in self._out]
        else:
            return self._out

    @staticmethod
    def run(data: npt.NDArray[np.float32], index: Union[List[datetime], List[str]], features: List[str],
            previous_data: Optional[npt.NDArray[np.float32]] = None, **params) -> npt.NDArray[np.float32]:
        """
        This method can be overridden to implement the indicator logic.  This method is called when the callback method
        is not set.
        :param data: The input data as a 2d numpy array.
        :param index: The index of the input data.  (datetime if not compiled with numba or string if compiled
        :param features: The features of the input data
        :param previous_data: An array of the previous indicator results.  The recent points in the array will be nan.
                        These are the values to fill.  This parameter is useful when streaming the indicator
                        (more compute efficient, but not all indicators support it.)
        :param params: Any parameters set by the user
        :return: A 2D array with the indicator results.  Must be the same length as the input data.
        """
        raise NotImplementedError("This method must be implemented in a subclass, or you must set the callback.")


    def __str__(self):
        return f"{self._name}({', '.join([f'{k}={v}' for k, v in self.params.items()])})"


    def __repr__(self):
        return f"Indicator({self._name})"

    @property
    def name(self) -> str:
        """
        :return: A unique name for the indicator.  (The name of the function and its identifier if different from 0)
        """
        if self._id == 0:
            return self._name
        else:
            return f"{self._name}_{self._id}"

    @property
    def type_name(self) -> str:
        """
        :return: The name of the indicator type
        """
        return self._name

    @property
    def identifier(self) -> int:
        return self._id

    def set_id(self, new: int):
        """
        Set the identifier of the indicator.  Used to identify the indicator in the indicator set when there are multiple
        indicators with the same name.
        :param new: The new id
        :return: None
        """
        self._id = new


"""
This module contains multiple helper function to execute a backtest.
"""
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

def crossover(series1: pd.Series, series2: pd.Series):
    """
    Check if series1 crosses over series2
    :param series1: Any pandas series with any index
    :param series2: Any pandas series with any index
    :return: True if series1 crosses over series2, False otherwise
    """
    return (series1.iloc[-1] > series2.iloc[-1]) & (series1.iloc[-2] < series2.iloc[-2])

def crossunder(series1: pd.Series, series2: pd.Series):
    """
    Check if series1 crosses under series2
    :param series1: Any pandas series with any index
    :param series2: Any pandas series with any index
    :return: True if series1 crosses under series2, False otherwise
    """
    return (series1.iloc[-1] < series2.iloc[-1]) & (series1.iloc[-2] > series2.iloc[-2])

def descending(series: pd.Series, lookback: int = 1):
    """
    Check if series is **continuously** descending.

    ## Example:

    >>> series = pd.Series([10, 7, 8, 7, 5])
    >>> descending(series, lookback=3) # True
    >>> descending(series, lookback=5) # False

    :param series: Any pandas series with any index
    :param lookback: Number of previous values to consider.  It is also the number of values that needs to be
    continuously descending.
    :return: True if series is continuously descending, False otherwise
    """
    a = series.to_numpy()
    delta = a[1:] - a[:-1]
    return (delta < 0)[-lookback:].all()

def ascending(series: pd.Series, lookback: int = 1):
    """
    Check if series is **continuously** ascending.

    ## Example:

    >>> series = pd.Series([1, 4, 3, 6, 7])
    >>> ascending(series, lookback=3) # True
    >>> ascending(series, lookback=5) # False

    :param series: Any pandas series with any index
    :param lookback: Number of previous values to consider.  It is also the number of values that needs to be
    continuously descending.
    :return: True if series is continuously ascending, False otherwise
    """
    a = series.to_numpy()
    delta = a[1:] - a[:-1]
    return (delta > 0)[-lookback:].all()
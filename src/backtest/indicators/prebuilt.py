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
from .indicator import Indicator
from typing import List
from datetime import datetime
import pandas as pd
try:
    import talib as ta
    TA_ENABLED = True
except ImportError:
    TA_ENABLED = False
import numpy as np

@Indicator(out_feat=["ADX"], period=int)
def ADX(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.ADX(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["ADXR"], period=int)
def ADXR(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
         period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.ADXR(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["APO"], fastperiod=int, slowperiod=int, matype=int)
def APO(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.APO(data[:, 4], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)[:, np.newaxis]

@Indicator(out_feat=["aroondown", "aroonup"], period=int)
def AROON(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
          period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    aroon_down, aroon_up = ta.AROON(data[:, 1], data[:, 2], timeperiod=period)
    return np.concatenate([aroon_down[:, np.newaxis], aroon_up[:, np.newaxis]], axis=1)


@Indicator(out_feat=["AROONOSC"], period=int)
def AROONOSC(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
             period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.AROONOSC(data[:, 1], data[:, 2], timeperiod=period)[:, np.newaxis]


@Indicator(out_feat=["BOP"])
def BOP(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.BOP(data[:, 0], data[:, 1], data[:, 2], data[:, 3])[:, np.newaxis]

@Indicator(out_feat=["CCI"], period=int)
def CCI(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.CCI(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["CMO"], period=int)
def CMO(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.CMO(data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["DX"], period=int)
def DX(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
       period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.DX(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["SMA"], period=int)
def SMA(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        period: int = 10) -> np.ndarray:
    """
    This SMA indicator support the streaming feature.  If the previous_data is not None, the indicator will fill the
    missing values in the previous_data and return it.  If the previous_data is None, it will calculate the indicator
    for the whole data.
    """
    if previous_data is None:
        return ta.SMA(data[:, 3], timeperiod=period)[:, np.newaxis]
    else:
        prev_sma = previous_data[:, 0]
        flipped_idx = np.flip(np.isnan(prev_sma)).argmin()
        idx = len(prev_sma) - flipped_idx
        if idx > period:
            prev_sma[idx:] = ta.SMA(data[idx-period:, 3], timeperiod=period)[period:]
            return prev_sma[:, np.newaxis]
        else:
            return ta.SMA(data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["EMA"], period=int)
def EMA(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
        period: int = 10) -> np.ndarray:
    """
    This EMA indicator support the streaming feature.  If the previous_data is not None, the indicator will fill the
    missing values in the previous_data and return it.  If the previous_data is None, it will calculate the indicator
    for the whole data.
    """
    if previous_data is None:
        return ta.EMA(data[:, 3], timeperiod=period)[:, np.newaxis]
    else:
        prev_sma = previous_data[:, 0]
        flipped_idx = np.flip(np.isnan(prev_sma)).argmin()
        idx = len(prev_sma) - flipped_idx
        if idx > period:
            prev_sma[idx:] = ta.EMA(data[idx-period:, 3], timeperiod=period)[period:]
            return prev_sma[:, np.newaxis]
        else:
            return ta.EMA(data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["MACD", "MACD_SIGNAL", "MACD_HIST"], fastperiod=int, slowperiod=int, signalperiod=int)
def MACD(data: np.ndarray, index: List[datetime], features: List[str], previous_data: np.ndarray,
         fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    macd, signal, hist = ta.MACD(data[:, 3], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return np.concatenate([macd[:, np.newaxis], signal[:, np.newaxis], hist[:, np.newaxis]], axis=1)
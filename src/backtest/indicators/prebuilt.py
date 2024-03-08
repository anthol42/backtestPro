from indicator import Indicator
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
def ADX(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.ADX(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["ADXR"], period=int)
def ADXR(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.ADXR(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["APO"], fastperiod=int, slowperiod=int, matype=int)
def APO(data: np.ndarray, index: List[datetime], features: List[str],
        fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.APO(data[:, 4], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)[:, np.newaxis]

@Indicator(out_feat=["aroondown", "aroonup"], period=int)
def AROON(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    aroon_down, aroon_up = ta.AROON(data[:, 1], data[:, 2], timeperiod=period)
    return np.concatenate([aroon_down[:, np.newaxis], aroon_up[:, np.newaxis]], axis=1)


@Indicator(out_feat=["AROONOSC"], period=int)
def AROONOSC(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.AROONOSC(data[:, 1], data[:, 2], timeperiod=period)[:, np.newaxis]


@Indicator(out_feat=["BOP"])
def BOP(data: np.ndarray, index: List[datetime], features: List[str]) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.BOP(data[:, 0], data[:, 1], data[:, 2], data[:, 3])[:, np.newaxis]

@Indicator(out_feat=["CCI"], period=int)
def CCI(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.CCI(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["CMO"], period=int)
def CMO(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.CMO(data[:, 3], timeperiod=period)[:, np.newaxis]

@Indicator(out_feat=["DX"], period=int)
def DX(data: np.ndarray, index: List[datetime], features: List[str], period: int = 14) -> np.ndarray:
    """
    Data is a 2D numpy array with the shape (n, f) where n is the number of observations.  The first 4 columns of the
    array are the OHLC data.
    """
    return ta.DX(data[:, 1], data[:, 2], data[:, 3], timeperiod=period)[:, np.newaxis]
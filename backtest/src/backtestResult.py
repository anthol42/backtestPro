from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from .broker import Broker
from .account import Account
from .metadata import Metadata
import pandas as pd
from enum import Enum
from typing import Union
import json
from pathlib import PurePath
from utils import *
import backtest

class Period(Enum):
    Yearly = 1
    QUARTERLY = 2
    Monthly = 3
    Weekly = 4
    Daily = 5
    HOURLY = 6


class BackTestResult:
    def __init__(self, strategy_name: str, metadata: Metadata, start: datetime, end: datetime, intial_cash: float,
                 market_index: npt.NDArray[np.float64], broker: Broker, account: Account,
                 risk_free_rate: float = 1.5):
        """
        This class is used to store the result of a backtest.  It contains all the information about the backtest
        :param strategy_name: The name of the strategy
        :param metadata: The metadata object
        :param start: Start date of the backtest data
        :param end: End date of the backtest data
        :param intial_cash: The initial cash of the account
        :param market_index: The market index time series (Price/score evolution across time)
        :param broker: The broker object used to backtest the strategy
        :param account: The account object used to backtest the strategy
        :param risk_free_rate: The risk free rate used to compute the ratios [%].  (Usually the estimated policy rate)
        """

        equity_history = np.array([stepState.worth for stepState in broker.historical_states])
        timestamps = np.array([stepState.timestamp for stepState in broker.historical_states])

        # About the backtest (Including steps to reproduce, environment conditions, etc.)
        self.metadata = metadata
        self.strategy_name = strategy_name

        # Unique values
        self.start = start
        self.end = end
        self.duration = end - start    # Total duration of simulation
        self.exposure_time = broker.exposure_time
        self.equity_final = equity_history[-1]
        self.equity_peak = equity_history.max()
        self.returns = 100 * ((equity_history[-1] - intial_cash) / intial_cash).item
        self.market_index = market_index
        self.index_returns = 100 * (market_index[-1] - market_index[0]) / market_index[0]    # Buy and hold
        self.annual_returns = self._get_annual_returns(self.duration, self.returns)
        self.sharp_ratio = (self.annual_returns - risk_free_rate) / np.std(equity_history)
        self.sortino_ratio = self.compute_sortino_ratio(risk_free_rate)

        # Now, calculate the drawdown
        drawdown = self.get_drawdown(equity_history, timestamps)
        drawdown_series = pd.Series(data=drawdown, index=pd.DatetimeIndex(timestamps))
        yearly_drawdown = drawdown_series.resample("Y").min()
        yearly_drawdown.index = yearly_drawdown.index.year
        self.max_drawdown = drawdown.max()
        self.avg_drawdown = drawdown.mean()
        self.calmar_ratio = self.compute_calmar_ratio(yearly_drawdown)
        self.num_trades = broker.portfolio.get_trade_count(exit_only=False)

        # Get trade stats
        trade_stats = broker.portfolio.get_trade_stats()
        self.win_rate = trade_stats["win_rate"]
        self.best_trade = trade_stats["best_trade"]
        self.worst_trade = trade_stats["worst_trade"]
        self.avg_trade = trade_stats["avg_trade"]
        self.max_trade_duration = trade_stats["max_trade_duration"]
        self.avg_trade_duration = trade_stats["avg_trade_duration"]
        self.min_trade_duration = trade_stats["min_trade_duration"]
        self.profit_factor = trade_stats["profit_factor"]
        self.sqn = trade_stats["SQN"]

        # Series
        self.broker = broker
        self.account_state = account.get_state()
        self.broker_state = broker.get_state()
        self.equity_history = equity_history.tolist()

    @staticmethod
    def _get_annual_returns(duration: timedelta, returns: float):
        """
        Get the annual returns of a strategy
        :param duration: The duration of the simulation
        :param returns: The overall returns of the strategy
        :return: The annual returns of the strategy (esperance) in percentage
        """
        duration_in_years = duration.total_seconds() / (365*86_400)
        return 100 * np.exp(np.log(returns) / duration_in_years)


    def get_ohlc(self, period: Period):
        """
        Return a OHLC dataframe of the equity history corresponding to the period.
        :param period: The period to use to make the OHLC dataframe
        :return: The OHLC dataframe
        """
        equity_data = np.array([stepState.worth for stepState in self.broker.historical_states]).reshape(-1, 1)
        equity_index = pd.DatetimeIndex(np.array([stepState.timestamp for stepState in self.broker.historical_states]))
        equity_series = pd.DataFrame(equity_data, index=equity_index)

        if period == Period.Yearly:
            # Copilot generated, need to check if it works
            return equity_series.resample("Y").ohlc()
        elif period == Period.QUARTERLY:
            return equity_series.resample("Q").ohlc()
        elif period == Period.Monthly:
            return equity_series.resample("M").ohlc()
        elif period == Period.Daily:
            return equity_series.resample("D").ohlc()
        elif period == Period.Weekly:
            return equity_series.resample("W").ohlc()

    def compute_sortino_ratio(self, risk_free_rate) -> float:
        """
        Compute the sortino ratio of the strategy
        :param risk_free_rate: The risk free rate or MAR (Minimum acceptable return)
        :return: The sortino ratio of the strategy
        """
        equity_ohlc = self.get_ohlc(Period.Yearly)
        diff = equity_ohlc["close"].diff()
        diff[0] = equity_ohlc["close"].iloc[0] - equity_ohlc["open"].iloc[0]
        diff_percentage = 100 * diff / equity_ohlc["close"].shift(1)
        downside = diff_percentage[diff_percentage < 0].to_numpy()
        downside_deviation = (downside ** 2).sum() / len(diff_percentage)
        return (self.annual_returns - risk_free_rate) / downside_deviation

    def compute_calmar_ratio(self, yearly_max_drawdown: pd.Series) -> float:
        """
        Compute the calmar ratio of the strategy
        :param yearly_max_drawdown: The yearly maximum drawdown of the strategy
        :return: The calmar ratio of the strategy
        """
        equity_ohlc = self.get_ohlc(Period.Yearly)
        diff = equity_ohlc["close"].diff()
        diff_percentage = 100 * diff / equity_ohlc["close"].shift(1)
        diff_percentage[0] = (equity_ohlc["close"].iloc[0] - equity_ohlc["open"].iloc[0]) / equity_ohlc["open"].iloc[0]
        assert diff.shape == yearly_max_drawdown.shape
        return (diff_percentage / yearly_max_drawdown).mean()


    def get_drawdown(self, equity_history: np.ndarray, equity_timestamps: np.ndarray,
                     window: Union[timedelta, int] = '1y') -> npt.NDArray[np.float32]:
        """
        Get the maximum drawdown of the strategy for each timestep (Causally - so looking back inn time)
        :param equity_history: The equity history of the strategy (Worth of the portoflio evolution over time)
        :param equity_timestamps: The timestamps of the equity history
        :param window: The lookback window to use to compute the drawdown
        :return: A time series of the maximum drawdown of the strategy for each time steps
        """
        if isinstance(window, timedelta):
            df = pd.DataFrame(data=[equity_history], columns=["Worth"], index=pd.DatetimeIndex(equity_timestamps))
            start = df.index[0]
            window_arr = df["Worth"].loc[:start + window].to_numpy()
            window = len(window_arr)
        start_maxes = np.maximum.accumulate(equity_history[:window - 1])
        maxes = np.concatenate((start_maxes, self.strided_arr(equity_history, window).max(axis=1)))
        drawdown_continum = (window - maxes) / maxes
        start_drawdown = np.minimum.accumulate(drawdown_continum[:window - 1])
        drawdown = np.concatenate((start_drawdown, self.strided_arr(drawdown_continum, window).min(axis=1)))
        return drawdown

    @staticmethod
    def strided_arr(a: np.ndarray, window: int):
        """
        Create a strided array to compute the rolling window of an array
        :param a: The array
        :param window: The window size
        :return: The strided array.  Shape(len(a) - window + 1, window)
        """
        nrows = ((a.size - window)) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, window), strides=(n, n))

    def get_state(self) -> dict:
        return {
            "metadata": self.metadata.export(),
            "stats":{
                "strategy_name": self.strategy_name,
                "start": str(self.start),
                "end": str(self.end),
                "duration": str(self.duration),
                "exposure_time": str(self.exposure_time),
                "equity_final": self.equity_final,
                "equity_peak": self.equity_peak,
                "returns": self.returns,
                "index_returns": self.index_returns,
                "annual_returns": self.annual_returns,
                "sharp_ratio": self.sharp_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "avg_drawdown": self.avg_drawdown,
                "calmar_ratio": self.calmar_ratio,
                "num_trades": self.num_trades,
                "win_rate": self.win_rate,
                "best_trade": self.best_trade,
                "worst_trade": self.worst_trade,
                "avg_trade": self.avg_trade,
                "max_trade_duration": self.max_trade_duration,
                "avg_trade_duration": self.avg_trade_duration,
                "min_trade_duration": self.min_trade_duration,
                "profit_factor": self.profit_factor,
                "sqn": self.sqn,
            },
            "run_states": {
                "run_timestamp": str(datetime.now()),
                "broker": self.broker_state,
                "account": self.account_state,
                "equity_history": self.equity_history
            },
            "sys_info":{
                "software": {
                    "python": get_py_version(),
                    "finBacktest": backtest.__version__
                },
                "platform": get_platform(),
                "hardware": get_hardware()
            }
        }

    def save(self, path: str):
        """
        Save the backtest result to a JSON file with the .bktst extension if none is provided.
        About the .bktst extension:
            This is the fileformat for a backtest result.  It is a JSON file that contains all the information about the
            backtest.  It is a human readable file that can be used to share backtest results with other people.
            Also, it can be loaded by this module so that the backtest result can be analyzed later.
        :param path: The path to save the backtest result.  If no extension is provided, the .bktst extension will be added.
        :return: None
        """
        extension = PurePath(path).suffix
        if extension == "":
            path = path + ".bktst"
        elif "/" in extension:
            path = path + ".bktst"
        with open(path, "w") as f:
            json.dump(self.get_state(), f, indent=4)

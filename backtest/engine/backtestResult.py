from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from .broker import Broker
from .account import Account
from .metadata import Metadata
from .tsData import TSData
from .analyser import Analyser
import pandas as pd
from enum import Enum
from typing import Union, List, Dict
import json
from pathlib import PurePath
from .utils import *
import backtest
from typing import Optional

class Period(Enum):
    YEARLY = 1
    QUARTERLY = 2
    MONTHLY = 3
    WEEKLY = 4
    DAILY = 5
    HOURLY = 6


class BackTestResult:
    def __init__(self, strategy_name: str, metadata: Metadata, start: datetime, end: datetime, intial_cash: float,
                 market_index: Optional[npt.NDArray[np.float64]], broker: Broker, account: Account,
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
        self.initial_cash = intial_cash
        self.historical_states = broker.historical_states

        # Unique values
        self.start = start
        self.end = end
        self.duration = end - start    # Total duration of simulation
        self.exposure_time = broker.exposure_time
        self.equity_final = equity_history[-1]
        self.equity_peak = equity_history.max()
        self.returns = 100 * ((equity_history[-1] - intial_cash) / intial_cash).item()
        self.market_index = market_index
        if market_index is not None:
            self.index_returns = 100 * (market_index[-1] - market_index[0]) / market_index[0]    # Buy and hold
        else:
            self.index_returns = None
        self.annual_returns = self._get_annual_returns(self.duration, self.returns)
        self.sharp_ratio = self.compute_sharp_ratio(risk_free_rate)
        self.sortino_ratio = self.compute_sortino_ratio(risk_free_rate)

        # Now, calculate the drawdown
        drawdown = self.get_drawdown(equity_history, timestamps)
        drawdown_series = pd.Series(data=drawdown, index=pd.DatetimeIndex(timestamps))
        yearly_drawdown = drawdown_series.resample("YE", closed='left', label='left').min()
        yearly_drawdown.index = yearly_drawdown.index.year
        self.max_drawdown = -100 * drawdown.min()
        self.avg_drawdown = -100 * drawdown.mean()
        self.calmar_ratio = self.compute_calmar_ratio(yearly_drawdown)
        self.num_trades = broker.portfolio.get_trade_count(exit_only=False)
        self.num_exits = broker.portfolio.get_trade_count(exit_only=True)

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

    def __str__(self):
        return (f"Backtest results for {self.strategy_name} from {self.start} to {self.end}:\n"
                f"\tDuration:                  {self.duration}\n"
                f"\tExposure time [days]:      {self.exposure_time}\n"
                f"\tEquity final [$]:          {self.equity_final}\n"
                f"\tEquity peak [$]:           {self.equity_peak}\n"
                f"\tReturns [%]:               {self.returns}\n"
                f"\tIndex Returns [%]:         {self.index_returns}\n"
                f"\tAnnual returns [%]:        {self.annual_returns}\n"
                f"\tSharp ratio:               {self.sharp_ratio}\n"
                f"\tSortino ratio:             {self.sortino_ratio}\n"
                f"\tMax drawdown [%]:          {self.max_drawdown}\n"
                f"\tAvg drawdown [%]:          {self.avg_drawdown}\n"
                f"\tCalmar ratio:              {self.calmar_ratio}\n"
                f"\tNum trades:                {self.num_trades}\n"
                f"\tNum exits:                 {self.num_exits}\n"
                f"\tWin rate [%]:              {self.win_rate}\n"
                f"\tBest trade [%]:            {self.best_trade}\n"
                f"\tWorst trade [%]:           {self.worst_trade}\n"
                f"\tAvg trade [%]:             {self.avg_trade}\n"
                f"\tMax trade duration [days]: {self.max_trade_duration}\n"
                f"\tAvg trade duration [days]: {self.avg_trade_duration}\n"
                f"\tMin trade duration [days]: {self.min_trade_duration}\n"
                f"\tProfit factor:             {self.profit_factor}\n"
                f"\tSQN:                       {self.sqn}\n")

    @staticmethod
    def _get_annual_returns(duration: timedelta, returns: float):
        """
        Get the annual returns of a strategy
        :param duration: The duration of the simulation
        :param returns: The overall returns of the strategy
        :return: The annual returns of the strategy (esperance) in percentage
        """
        duration_in_years = duration.total_seconds() / (365*86_400)
        return 100 * np.exp(np.log(returns / 100 + 1) / duration_in_years) - 100


    def get_ohlc(self, period: Period):
        """
        Return a OHLC dataframe of the equity history corresponding to the period.
        :param period: The period to use to make the OHLC dataframe
        :return: The OHLC dataframe
        """
        equity_data = np.array([stepState.worth for stepState in self.historical_states]).reshape(-1, 1)
        equity_index = pd.DatetimeIndex(np.array([stepState.timestamp for stepState in self.historical_states]))
        equity_series = pd.DataFrame(equity_data, index=equity_index)

        if period == Period.YEARLY:
            # Copilot generated, need to check if it works
            out = equity_series.resample("YE", closed='left', label='left').ohlc()
        elif period == Period.QUARTERLY:
            out = equity_series.resample("Q", closed='left', label='left').ohlc()
        elif period == Period.MONTHLY:
            out = equity_series.resample("M", closed='left', label='left').ohlc()
        elif period == Period.DAILY:
            out = equity_series.resample("D", closed='left', label='left').ohlc()
        elif period == Period.WEEKLY:
            out = equity_series.resample("W", closed='left', label='left').ohlc()
        out.columns = out.columns.droplevel()
        return out.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    def compute_sharp_ratio(self, risk_free_rate) -> float:
        """
        Compute the sharp ratio of the strategy on a weekly basis
        :param risk_free_rate: The risk free rate or MAR (Minimum acceptable return)
        :return: The sharp ratio of the strategy
        """
        equity_ohlc = self.get_ohlc(Period.WEEKLY)
        diff = equity_ohlc["Close"].diff()
        diff.iloc[0] = equity_ohlc["Close"].iloc[0] - equity_ohlc["Open"].iloc[0]
        diff_ratio = diff / equity_ohlc["Close"].shift(1)
        diff_ratio.iloc[0] = (equity_ohlc["Close"].iloc[0] - equity_ohlc["Open"].iloc[0]) / \
                                  equity_ohlc["Open"].iloc[0]
        if diff.std() == 0:
            return None
        else:
            std = np.sqrt(52)*diff_ratio.std()    # Annualize the std
            return (self.annual_returns - risk_free_rate) / (100 * std)
    def compute_sortino_ratio(self, risk_free_rate) -> float:
        """
        Compute the sortino ratio of the strategy on a weekly basis
        :param risk_free_rate: The risk free rate or MAR (Minimum acceptable return)
        :return: The sortino ratio of the strategy
        """
        equity_ohlc = self.get_ohlc(Period.WEEKLY)
        diff = equity_ohlc["Close"].diff()
        diff.iloc[0] = equity_ohlc["Close"].iloc[0] - equity_ohlc["Open"].iloc[0]
        diff_percentage = (diff / equity_ohlc["Close"].shift(1)) - risk_free_rate / 100
        downside = diff_percentage[diff_percentage < 0].to_numpy()
        annualized_downside_deviation = np.sqrt(52 * (downside ** 2).sum() / (len(diff_percentage) - 1))
        if annualized_downside_deviation == 0:
            return None
        else:
            return (self.annual_returns - risk_free_rate) / (100 * annualized_downside_deviation)

    def compute_calmar_ratio(self, yearly_max_drawdown: pd.Series) -> Optional[float]:
        """
        Compute the calmar ratio of the strategy
        :param yearly_max_drawdown: The yearly maximum drawdown of the strategy
        :return: The calmar ratio of the strategy
        """
        equity_ohlc = self.get_ohlc(Period.YEARLY)
        diff = equity_ohlc["Close"].diff()
        diff_percentage = 100 * diff / equity_ohlc["Close"].shift(1)
        diff_percentage.iloc[0] = 100 * (equity_ohlc["Close"].iloc[0] - equity_ohlc["Open"].iloc[0]) / equity_ohlc["Open"].iloc[0]
        assert diff.shape == yearly_max_drawdown.shape
        if (yearly_max_drawdown == 0).any():
            return None
        else:
            ts_ratio = diff_percentage.to_numpy() / (-100 * yearly_max_drawdown.to_numpy())
            return ts_ratio.mean()


    def get_drawdown(self, equity_history: np.ndarray, equity_timestamps: np.ndarray,
                     window: Union[timedelta, int] = timedelta(days=365)) -> npt.NDArray[np.float32]:
        """
        Get the maximum drawdown of the strategy for each timestep (Causally - so looking back in time)
        :param equity_history: The equity history of the strategy (Worth of the portoflio evolution over time)
        :param equity_timestamps: The timestamps of the equity history
        :param window: The lookback window to use to compute the drawdown
        :return: A time series of the maximum drawdown of the strategy for each time steps
        """
        if isinstance(window, timedelta):
            df = pd.DataFrame(data=equity_history[:, np.newaxis], columns=["Worth"], index=pd.DatetimeIndex(equity_timestamps))
            start = df.index[0]
            window_arr = df["Worth"].loc[:start + window].to_numpy()
            window = len(window_arr)
        start_maxes = np.maximum.accumulate(equity_history[:window - 1])
        maxes = np.concatenate((start_maxes, self.strided_arr(equity_history, window).max(axis=1)))
        drawdown_continum = (equity_history - maxes) / maxes
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
                "equity_final": float(self.equity_final),
                "equity_peak": float(self.equity_peak),
                "returns": float(self.returns),
                "index_returns": float(self.index_returns) if self.index_returns is not None else None,
                "annual_returns": float(self.annual_returns) if self.annual_returns is not None else None,
                "sharp_ratio": float(self.sharp_ratio) if self.sharp_ratio is not None else None,
                "sortino_ratio": float(self.sortino_ratio) if self.sortino_ratio is not None else None,
                "max_drawdown": float(self.max_drawdown) if self.max_drawdown is not None else None,
                "avg_drawdown": float(self.avg_drawdown) if self.avg_drawdown is not None else None,
                "calmar_ratio": float(self.calmar_ratio) if self.calmar_ratio is not None else None,
                "num_trades": float(self.num_trades),
                "num_exits": float(self.num_exits),
                "win_rate": float(self.win_rate) if self.win_rate is not None else None,
                "best_trade": float(self.best_trade) if self.best_trade is not None else None,
                "worst_trade": float(self.worst_trade) if self.worst_trade is not None else None,
                "avg_trade": float(self.avg_trade) if self.avg_trade is not None else None,
                "max_trade_duration": float(self.max_trade_duration) if self.max_trade_duration is not None else None,
                "avg_trade_duration": float(self.avg_trade_duration) if self.avg_trade_duration is not None else None,
                "min_trade_duration":float(self.min_trade_duration) if self.min_trade_duration is not None else None,
                "profit_factor": float(self.profit_factor) if self.profit_factor is not None else None,
                "sqn": float(self.sqn) if self.sqn is not None else None,
            },
            "run_states": {
                "run_timestamp": str(datetime.now()),
                "broker": self.broker_state,
                "account": self.account_state,
                "equity_history": self.equity_history,
                "market_index": self.market_index.tolist() if self.market_index is not None else None
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

    @classmethod
    def load_state(cls, data: dict):
        """
        Load a backtest result from a state dictionary
        :param data: The state dictionary
        :return: The backtest result
        """
        metadata = Metadata.load(data["metadata"])
        account = Account.load_state(data["run_states"]["account"])
        broker = Broker.load_state(data["run_states"]["broker"], account)
        self = cls(data["stats"]["strategy_name"], metadata, datetime.fromisoformat(data["stats"]["start"]),
                   datetime.fromisoformat(data["stats"]["end"]), data["metadata"]["backtest_parameters"]["initial_cash"],
                   np.array(data["run_states"]["market_index"]), broker, account,
                   risk_free_rate=data["metadata"]["backtest_parameters"]["risk_free_rate"])

        self.equity_history = data["run_states"]["equity_history"]
        self.start = datetime.fromisoformat(data["stats"]["start"])
        self.end = datetime.fromisoformat(data["stats"]["end"])
        self.duration = self.end - self.start
        self.exposure_time = timedelta(seconds=float(data["stats"]["exposure_time"]))
        self.equity_final = data["stats"]["equity_final"]
        self.equity_peak = data["stats"]["equity_peak"]
        self.returns = data["stats"]["returns"]
        self.index_returns = data["stats"]["index_returns"]
        self.annual_returns = data["stats"]["annual_returns"]
        self.sharp_ratio = data["stats"]["sharp_ratio"]
        self.sortino_ratio = data["stats"]["sortino_ratio"]
        self.max_drawdown = data["stats"]["max_drawdown"]
        self.avg_drawdown = data["stats"]["avg_drawdown"]
        self.calmar_ratio = data["stats"]["calmar_ratio"]
        self.num_trades = data["stats"]["num_trades"]
        self.num_exits = data["stats"]["num_exits"]
        self.win_rate = data["stats"]["win_rate"]
        self.best_trade = data["stats"]["best_trade"]
        self.worst_trade = data["stats"]["worst_trade"]
        self.avg_trade = data["stats"]["avg_trade"]
        self.max_trade_duration = data["stats"]["max_trade_duration"]
        self.avg_trade_duration = data["stats"]["avg_trade_duration"]
        self.min_trade_duration = data["stats"]["min_trade_duration"]
        self.profit_factor = data["stats"]["profit_factor"]
        self.sqn = data["stats"]["sqn"]
        self.historical_states = broker.historical_states
        return self

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

    @classmethod
    def load(cls, path: str):
        """
        Load a backtest result from a JSON file
        :param path: The path to the backtest result file
        :return: The backtest result
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.load_state(data)


    def toAnalyser(self, data: List[Dict[str, TSData]]):
        """
        Convert the backtest result to an analyser object to visualize the backtest results
        :return: The analyser object
        """
        return Analyser(data, self.get_state())

from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
import pandas as pd
from enum import Enum
from .state_signals import StateSignals
from typing import Union, List
from typing import Optional

class Period(Enum):
    YEARLY = 1
    QUARTERLY = 2
    MONTHLY = 3
    WEEKLY = 4
    DAILY = 5
    HOURLY = 6


class StatCalculator:
    """
    Real time version of the BacktestResults object.  In calculates the stats based on a one year moving window.
    """
    def __init__(self, state: StateSignals):
        if state.backtest_params is not None and "risk_free_rate" in state.backtest_params:
            risk_free_rate = state.backtest_params["risk_free_rate"]
        else:
            risk_free_rate = 1.5  # Default risk free rate

        # Prepare data
        cutoff = state.timestamp - pd.Timedelta(days=365)
        portfolio_cutoff_worth = np.array([s.worth for s in state.broker.historical_states if s.timestamp > cutoff])
        equity_history = np.array([s.worth for s in state.broker.historical_states])
        security_names, data = self.prepare_data(state)
        current_worth = state.broker.get_worth(security_names, data)
        portfolio_cutoff_worth = np.append(portfolio_cutoff_worth, current_worth)
        equity_history = np.append(equity_history, current_worth)
        idx_name = list(state.index_data.keys())[0]
        index_worth = state.index_data[idx_name].data["Close"].loc[cutoff:state.timestamp].values
        portfolio_cutoff_timestamps = pd.DatetimeIndex(
            [s.timestamp for s in state.broker.historical_states if s.timestamp > cutoff])
        equity_timestamps = pd.DatetimeIndex([s.timestamp for s in state.broker.historical_states])
        portfolio_cutoff_timestamps = portfolio_cutoff_timestamps.append(pd.DatetimeIndex([state.timestamp]))
        equity_timestamps = equity_timestamps.append(pd.DatetimeIndex([state.timestamp]))



        self.equity_history = equity_history
        self.timestamps = equity_timestamps

        self.initial_cash = state.initial_cash
        # Get yearly added cash
        self.added_cash = state.cash_controller.total_deposited
        self.yearly_added_cash = state.cash_controller.yearly_variation(state.timestamp)

        # All-time
        self.start = equity_timestamps[0]
        self.end = equity_timestamps[-1]
        self.duration = self.end - self.start    # Overall duration to calculate all time stats
        self.equity_final = equity_history[-1]
        self.equity_peak = equity_history.max()
        self.all_time_returns = 100 * ((equity_history[-1] - self.initial_cash - self.added_cash) / self.initial_cash).item()
        self.annual_returns = self._get_annual_returns(self.duration, self.all_time_returns)
        self.sharp_ratio = self.compute_sharp_ratio(risk_free_rate)
        self.sortino_ratio = self.compute_sortino_ratio(risk_free_rate)

        # Now, calculate the drawdown [All-Time]
        drawdown = self.get_drawdown(equity_history, equity_timestamps)
        drawdown_series = pd.Series(data=drawdown, index=pd.DatetimeIndex(equity_timestamps))
        yearly_drawdown = drawdown_series.resample("YE", closed='left', label='left').min()
        yearly_drawdown.index = yearly_drawdown.index.year
        self.max_drawdown = -100 * drawdown.min()
        self.avg_drawdown = -100 * drawdown.mean()
        self.calmar_ratio = self.compute_calmar_ratio(yearly_drawdown)

        # Yearly
        self.year_returns = 100 * (portfolio_cutoff_worth[-1] - self.yearly_added_cash) / portfolio_cutoff_worth[0] - 100
        self.market_index = index_worth
        if index_worth is not None:
            self.index_returns = 100 * (index_worth[-1] - index_worth[0]) / index_worth[0]    # For the yearly period
        else:
            self.index_returns = None
        self.num_trades = state.broker.portfolio.get_trade_count(exit_only=False, cutoff=cutoff)
        self.num_exits = state.broker.portfolio.get_trade_count(exit_only=True, cutoff=cutoff)

        # Get trade stats [ Yearly ]
        trade_stats = state.broker.portfolio.get_trade_stats(cutoff=cutoff)
        self.win_rate = trade_stats["win_rate"]
        self.best_trade = trade_stats["best_trade"]
        self.worst_trade = trade_stats["worst_trade"]
        self.avg_trade = trade_stats["avg_trade"]
        self.max_trade_duration = trade_stats["max_trade_duration"]
        self.avg_trade_duration = trade_stats["avg_trade_duration"]
        self.min_trade_duration = trade_stats["min_trade_duration"]
        self.profit_factor = trade_stats["profit_factor"]
        self.sqn = trade_stats["SQN"]

    @staticmethod
    def prepare_data(state: StateSignals) -> (List[str], np.ndarray):
        """
        Extracts the relevent up-to-date data from the state object and returns it as an array.
        """
        data = state.data[state.main_idx]
        security_names = list(data.keys())
        out = np.empty((len(security_names), 4), dtype=np.float32)    # Shape(n, 4) for OHLC
        for i, name in enumerate(security_names):
            out[i, 0] = data[name].data["Open"].iloc[-1]
            out[i, 1] = data[name].data["High"].iloc[-1]
            out[i, 2] = data[name].data["Low"].iloc[-1]
            out[i, 3] = data[name].data["Close"].iloc[-1]

        return security_names, out

    @staticmethod
    def _get_annual_returns(duration: timedelta, returns: float):
        """
        Get the annual returns of a strategy
        :param duration: The duration of the simulation
        :param returns: The overall returns of the strategy
        :return: The annual returns of the strategy (esperance) in percentage
        """
        duration_in_years = duration.total_seconds() / (365 * 86_400)
        return 100 * np.exp(np.log(returns / 100 + 1) / duration_in_years) - 100

    def get_ohlc(self, period: Period):
        """
        Return a OHLC dataframe of the equity history corresponding to the period.
        :param period: The period to use to make the OHLC dataframe
        :return: The OHLC dataframe
        """
        equity_data = self.equity_history.reshape(-1, 1)
        equity_index = self.timestamps
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
            std = np.sqrt(52) * diff_ratio.std()  # Annualize the std
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
        if len(diff_percentage) < 2:
            return None
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
        diff_percentage.iloc[0] = 100 * (equity_ohlc["Close"].iloc[0] - equity_ohlc["Open"].iloc[0]) / \
                                  equity_ohlc["Open"].iloc[0]
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
            df = pd.DataFrame(data=equity_history[:, np.newaxis], columns=["Worth"],
                              index=pd.DatetimeIndex(equity_timestamps))
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
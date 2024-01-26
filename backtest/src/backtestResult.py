from datetime import datetime, timedelta
import numpy as np
from .broker import Broker
import pandas as pd
from enum import Enum

class Period(Enum):
    Yearly = 1
    QUARTERLY = 2
    Monthly = 3
    Weekly = 4
    Daily = 5
    HOURLY = 6


class BackTestResult:
    def __init__(self, start: datetime, end: datetime,  exposure_time: timedelta, equity_final: float,
                 equity_peak: float, returns: float, index_returns: float,
                    sharp_ratio: float, sortino_ratio: float, calmar_ratio: float, max_drawdown: float,
                 broker: Broker,
                 risk_free_rate: float = 1.5):
        equity_history = np.array([stepState.worth for stepState in broker.historical_states])

        # About the backtest (Including steps to reproduce, environment conditions, etc.)
        self.metadata = None
        self.strategy_name = None

        # Unique values
        self.start = start
        self.end = end
        self.duration = end - start    # Total duration of simulation
        self.exposure_time = exposure_time
        self.equity_final = equity_final
        self.equity_peak = equity_peak
        self.returns = returns
        self.index_returns = index_returns    # Buy and hold
        self.annual_returns = self._get_annual_returns(self.duration, self.returns)
        self.sharp_ratio = (self.annual_returns - risk_free_rate) / np.std(equity_history)
        self.sortino_ratio = self.compute_sortino_ratio(risk_free_rate)
        self.calmar_ratio = None
        self.max_drawdown = None
        self.avg_drawdown = None
        self.num_trades = None
        self.win_rate = None
        self.best_trade = None
        self.worst_trade = None
        self.avg_trade = None
        self.max_trade_duration = None
        self.avg_trade_duration = None
        self.min_trade_duration = None
        self.profit_factor = None
        self.expectancy = None
        self.sqn = None

        # Series
        self.broker = broker
        self.portfolio_state = None
        self.account_state = None
        self.transactions = None
        self.equity_history = None
        self.margin_calls = None

    @staticmethod
    def _get_annual_returns(duration: timedelta, returns: float):
        """
        Get the annual returns of a strategy
        :param duration: The duration of the simulation
        :param returns: The overall returns of the strategy
        :return: The annual returns of the strategy (esperance)
        """
        duration_in_years = duration.total_seconds() / (365*86_400)
        return np.exp(np.log(returns) / duration_in_years)


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
        downside = diff[diff < 0].to_numpy()
        downside_deviation = (downside ** 2).sum() / len(diff)
        return (self.annual_returns - risk_free_rate) / downside_deviation
import pandas as pd
from typing import Optional
from .tsData import DividendFrequency

class Record:
    """
    This class contains information about a stock and is pass to the strategy.
    """
    def __init__(self, chart: Optional[pd.DataFrame], ticker: str, time_res: int, marginable: bool, shortable: bool,
                 div_freq: DividendFrequency,
                 short_rate: float,
                 next_tick: Optional[pd.Series] = None):
        """

        :param chart: The prepared chart of the stock
        :param ticker: The ticker of the stock
        :param time_res: The idx in the list of resolution of the backtest object of the time resolution of the record.
        :param marginable: Whether the stock is marginable
        :param shortable: Whether the stock is shortable
        :param div_freq: The dividend frequency of the stock
        :param short_rate: The short rate of the stock
        :param next_tick: The next tick of the stock
        """
        self.chart = chart
        # The strategy should not use this attribute.  It is only for the broker
        self._next_tick: pd.Series = next_tick
        self.short_rate = short_rate
        self.ticker = ticker
        self.time_res = time_res
        self.marginable = marginable
        self.shortable = shortable
        self.div_freq = div_freq
        self.has_dividends = div_freq != DividendFrequency.NO_DIVIDENDS

    @property
    def next_tick(self):
        """
        This method is used to get the next tick of the stock.
        :return: The next tick of the stock
        """
        return self._next_tick

    def __eq__(self, other):
        return self.chart.equals(other.chart) and self.ticker == other.ticker and self.time_res == other.time_res and \
                self.marginable == other.marginable and self.shortable == other.shortable and self.div_freq == other.div_freq and \
                self.short_rate == other.short_rate and self.next_tick.equals(other.next_tick)
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
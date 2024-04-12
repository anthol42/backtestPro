from backtest import Strategy, Backtest
from backtest.indicators import IndicatorSet, TA
from backtest.data import FetchCharts, ToTSData, Cache, PadNan
import backtest.engine.functional as F
from datetime import datetime

class MyStrategy(Strategy):
    def run(self, data, timestep):
        for ticker in data.main.tickers:
            chart = data.main[ticker].chart
            if len(chart) > 2 and F.crossover(chart["MACD"], chart["MACD_SIGNAL"]) and chart["MACD"].iloc[-1] < 0:
                if ticker not in self.broker.portfolio.long:
                    self.broker.buy_long(ticker, 500)
            if ticker in self.broker.portfolio.long and F.crossunder(chart["MACD"], chart["MACD_SIGNAL"]):
                self.broker.sell_long(ticker, 500)

# The magnificent 7 tickers
TICKERS = ["META", "AMZN", "AAPL", "NVDA", "GOOGL", "MSFT", "TSLA"]
data_pipeline = FetchCharts(TICKERS, auto_adjust=False) | PadNan() | ToTSData() | Cache()
bt = Backtest(data_pipeline.get(datetime(2010, 1, 1), datetime(2020, 1, 1)),
              strategy=MyStrategy(),
              indicators=IndicatorSet(TA.MACD()))
results = bt.run()
print(results)

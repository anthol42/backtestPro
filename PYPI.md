# Backtest-pro: feature rich backtesting framework

## What it is?
**backtest-pro** is a framework that provides a way to test complex strategies in an environment that is designed to 
look as much as possible to the real world.  This way, the historical results are more likely to reflect the
real world results. It is an **event-driven** backtesting framework that is designed to be as flexible as possible and
as complete as possible.  It supports **end-to-end** quant pipeline from data fetching to production release.  
**Also, it has the broader goal of becoming the most complete backtesting framework available for python finely tuned 
for professional applications.**

## Important notice
**backtest-pro** is still in development and is not ready for production use.  There may be bugs that could
make the results of the backtest invalid.  Always double-check the results with your own code.  If you find a bug, please
open an issue on the github page.  The api might also change without notice.


## Features
Here are just a few of the features that **backtest-pro** offers:
- **DataPiplines**
  - An easy to use data pipeline api that makes building a data pipeline a breeze.
  - A pipeline built with the api is easy to maintain and easy to understand.
  - Support caching for more efficient pipelines.
- **Backtest**
  - Backtest with a single or with multiple assets simultaneously
  - Feed a moving window to the strategy
  - Multiple time resolution simultaneously
  - Take into account:
    - Trading fees
    - Margin rates
    - Margin calls
    - Stock splits
    - Dividends
  - Records a lot of metrics for easier analysis and debugging.
- **Release**
  - Use the same code as used in the backtest in production.
  - Only a few lines of codes are necessary to build a production pipeline that can run on a server or locally.
  - Automatic reporting of the results using report builders.  (Html, pdf)
  - Easy integration with other services such as api for algorithmic trading.

## Installation
To install **backtest-pro**, you can use pip:
```commandline
pip install backtest-pro
```
There a few dependencies that are not installed by default.  They are:
- TA-Lib: A technical analysis library that is used to calculate technical indicators.
- Plotly: A plotting library that is used to render charts in the production of reports.
- WeasyPrint: A library that is used to convert html to pdf.  It is used to render the reports in pdf format.
- kaleido: An optional library of Plotly that is used to render the charts in the reports.
- schedule: A library that is used to schedule the run of the strategy in production.
- python-crontab: A library that is used to schedule the run of the strategy in production.

To install the dependencies, you can use the following command:
```commandline
pip install backtest-pro[optional]
```

## Installation from source
To install **backtest-pro** from source, you can clone the repository and install it using pip:
```commandline
git clone https://github.com/anthol42/backtestPro.git
```
Move to the cloned repository:
```commandline
cd backtestPro
```
Then, you can install with the following command:
```commandline
pip install .
```

## Example
```python
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

```
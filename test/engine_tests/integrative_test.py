import pandas as pd

from backtest import BackTest, Strategy, Metadata, TSData, DividendFrequency, Record, Records, RecordsBucket
from backtest.engine import CashController, BasicExtender
from datetime import datetime, timedelta
from typing import List, Tuple
from unittest import TestCase


class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self._cash = {
            "AAPL": 5000,
            "NVDA": 5000
        }
    def run(self, data: RecordsBucket, timestep: datetime):
        # print(f"Running strategy at {timestep}")
        for ticker, record in data.main:
            chart = record.chart
            if chart["MA_delta"].iloc[-1] > 0.001 and chart["MA_delta"].iloc[-2] < 0.:
                # Buy for half of our money reserved to the stock
                cash_amount = self._cash[ticker] / 2
                price = chart["Close"].iloc[-1]
                shares = cash_amount // price
                self.broker.buy_long(ticker, shares, 0)
                # print(f"Buying {shares} shares of {ticker} at market price -- {timestep}")

            if chart["MA_delta"].iloc[-1] < -0.001:
                # Sell all shares
                long, short = self.broker.portfolio[ticker]
                shares = long.amount if long is not None else 0
                if shares > 0:
                    self.broker.sell_long(ticker, shares, 0)
                    # print(f"Selling {shares} shares of {ticker} at market price -- {timestep}")

        # print(f"Data len: {data[-1]['NVDA'].chart.shape[0]}")

    def indicators(self, data: RecordsBucket, timestep: datetime):
        """
        This method will add a 7 days moving average columns called MA to the dataframes
        :param data: The stock data
        :param timestep: The current timestep (Datetime)
        :return: The extended data
        """
        for time_res, records in data:
            if time_res == timedelta(days=1):
                for ticker, record in records:
                    record.chart['MA'] = record.chart['Close'].rolling(window=7).mean()
                    record.chart['MA_delta'] = (record.chart['MA'] - record.chart['MA'].shift(1)) / record.chart['MA'].shift(1)
                records.update_features()
        return data

class MyCashController(CashController):
    def every_month(self, timestamp: datetime) -> Tuple[float, str]:
        return 1000, "Monthly deposit"


class TestIntegration(TestCase):
    def test_integration(self):
        # Create the metadata
        metadata = Metadata(description="Integration Test")
        data = [
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"), name="AAPL-1h"),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"), name="NVDA-1h")
            }
        ]
        backtest = BackTest(data, MyStrategy(), main_timestep=1, initial_cash=10_000, commission=10.,
                            metadata=metadata, margin_interest=10,
                            time_res_extender=BasicExtender("1d") + BasicExtender("1w"),
                            cash_controller=MyCashController())

        results = backtest.run()
        print(results)
        results.save("tmp.bcktst")


# NVDA: 5 shares at 452.81
# AAPL: 12  shares at 194.20
# NVDA: 5 shares at 483.57
# NVDA: 4 shares at 495.12

# End:
# NVDA price: 785.27 total: 10993.78, cost: 6658.38; final profit: 4335.4
# AAPL price: 184.48 total: 2213.76, cost: 2330.4; final profit: -116.8


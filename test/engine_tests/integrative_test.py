import pandas as pd

from backtest import BackTest, Strategy, Metadata, TSData, DividendFrequency, Record
from backtest.engine import CashController, BasicExtender
from datetime import datetime, timedelta
from typing import List
from unittest import TestCase


class MyStrategy(Strategy):
    def run(self, data: List[List[Record]], timestep: datetime):
        # print(f"Running strategy at {timestep}")
        pass

    def indicators(self, data: List[List[Record]], timestep: datetime):
        """
        This method will add a 7 days moving average columns called MA to the dataframes
        :param data: The stock data
        :param timestep: The current timestep (Datetime)
        :return: The extended data
        """
        for time_res in data:
            if self.available_time_res[time_res[0].time_res] == timedelta(days=1):
                for record in time_res:
                    record.chart['MA'] = record.chart['Close'].rolling(window=7).mean()
        return data

class MyCashController(CashController):
    def every_month(self, timestamp: datetime):
        self.account.deposit(1000, timestamp, comment="Monthly deposit")


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

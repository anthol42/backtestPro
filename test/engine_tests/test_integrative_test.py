import pandas as pd
import numpy as np
import numpy.typing as npt
from src.backtest import Backtest, Strategy, Metadata, TSData, DividendFrequency, RecordsBucket
from src.backtest.engine import CashControllerBase, BasicExtender
from src.backtest.indicators import Indicator, IndicatorSet, TA
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Union
from unittest import TestCase
from integration_src.strategy import ComplexGoodStrategy, WeekCashController, ComplexBadStrategy, BadCashController
import os
from pathlib import PurePath


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
            if chart["MA_delta(7)"].iloc[-1] > 0.001 and chart["MA_delta(7)"].iloc[-2] < 0.:
                # Buy for half of our money reserved to the stock
                cash_amount = self._cash[ticker] / 2
                price = chart["Close"].iloc[-1]
                shares = cash_amount // price
                self.broker.buy_long(ticker, shares, 0)

            if chart["MA_delta(7)"].iloc[-1] < -0.001:
                # Sell all shares
                long, short = self.broker.portfolio[ticker]
                shares = long.amount if long is not None else 0
                if shares > 0:
                    self.broker.sell_long(ticker, shares)


@Indicator(out_feat=["MA", "MA_delta"], period=int)
def MADerivative(data: npt.NDArray[np.float32], index: Union[List[datetime], List[str]], features: List[str],
            previous_data: Optional[npt.NDArray[np.float32]] = None, period: int = 7):
        data = pd.DataFrame(data, index=index, columns=features)
        ma = data["Close"].rolling(window=period).mean()
        delta = (ma - ma.shift(1)) / ma.shift(1)
        return np.array([ma.values, delta.values]).T


class MyCashController(CashControllerBase):
    def every_month(self, timestamp: datetime) -> Tuple[float, str]:
        return 1000, "Monthly deposit"


class TestIntegration(TestCase):
    def test_integration1(self):
        """
        This if everything runs smoothly without crash.
        """
        os.chdir(PurePath(__file__).parent)
        # Create the metadata
        metadata = Metadata(description="Integration Test")
        data = [
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"), name="AAPL-1h"),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"), name="NVDA-1h")
            }
        ]
        backtest = Backtest(data, MyStrategy(), main_timestep=1, initial_cash=10_000, commission=10.,
                            metadata=metadata, margin_interest=10,
                            time_res_extender=BasicExtender("1d") + BasicExtender("1w"),
                            cash_controller=MyCashController(), indicators=IndicatorSet(MADerivative(period=7)))

        results = backtest.run()


    def test_integration2(self):
        """
        This test will verify if the results are accurate
        """
        os.chdir(PurePath(__file__).parent)
        data = [
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"), name="AAPL-dh"),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"), name="NVDA-dh")
            }
        ]
        index = TSData(pd.read_csv("test_data/SPY_6mo_1d.csv", index_col="Date"), name="SPY",
                       div_freq=DividendFrequency.NO_DIVIDENDS, time_res=timedelta(days=1))

        backtest = Backtest(data, ComplexGoodStrategy(), initial_cash=100_000, commission=10., margin_interest=10,
                            default_short_rate=20., default_shortable=True, default_marginable=True,
                            cash_controller=WeekCashController(),
                            market_index=index)

        results = backtest.run()
        self.assertAlmostEqual(152_635.64, results.equity_final, delta=0.01)
        print(results)

    def test_integration3(self):
        """
        Test margin call, liquidation and bankruptcy
        """
        os.chdir(PurePath(__file__).parent)
        data = [
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"), name="AAPL-dh"),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"), name="NVDA-dh")
            }
        ]
        backtest = Backtest(data, ComplexBadStrategy(), initial_cash=100_000, commission=10., margin_interest=10,
                            default_short_rate=20., default_shortable=True, default_marginable=True,
                            cash_controller=BadCashController(),
                            min_maintenance_margin_short=25)

        results = backtest.run()
        self.assertAlmostEqual(1109.15, results.bankruptcy_amount, delta=0.01)
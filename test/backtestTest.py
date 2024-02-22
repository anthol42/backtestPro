from unittest import TestCase
import pandas as pd
from typing import Tuple
from backtest.src.account import Account
from backtest.src.broker import Broker
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta
from backtest.src.tsData import TSData
from backtest.src.strategy import Strategy
from typing import List, Dict, Type, Optional
from backtest.src.backtestResult import BackTestResult
from backtest.src.record import Record
from backtest.src.tsData import DividendFrequency
from tqdm import tqdm
import warnings
from backtest.src.metadata import Metadata
from backtest.src.cashController import CashController
from backtest.backtest import BackTest


class MyStrat(Strategy):
    def run(self, data: List[List[Record]], timestep: datetime):
        pass

class TestBacktest(TestCase):
    def test_reverse_split_norm(self):
        DATA = pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date")
        data = [
            {"AAPL": TSData(DATA, name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY)}
        ]
        bcktst = BackTest(data, MyStrat())
        # Make fake data with split

        hist = pd.DataFrame(data=[
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             [1.13, 1.09, 1.11, 0.86, 100.0, 0.5, 0.0],
             [1.0, 0.86, 0.95, 0.99, 100.0, 0.5, 2.0],
             [1.03, 0.89, 1.26, 1.25, 100.0, 0.5, 0.0],
             [0.98, 1.1, 0.84, 0.7, 100.0, 0.5, 0.0],
             [0.95, 0.98, 0.99, 0.99, 100.0, 0.5, 5.0],
             [1.1, 1.0, 1.07, 1.01, 100.0, 0.5, 0.0],
             [0.98, 1.07, 1.09, 0.96, 100.0, 0.5, 0.0],
             [1.03, 1.15, 0.98, 1.09, 100.0, 0.5, 0.0],
             [1.02, 1.0, 1.07, 0.99, 100.0, 0.5, 0.0]
        ], columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
            index=pd.date_range(start="2021-01-01", periods=10, freq="D"),
        dtype=np.float64)
        bcktst._reverse_split_norm(hist)

        expected = np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [11.3, 10.9, 11.1, 8.6, 10.0, 0.05, 0.0],
                    [10.0, 8.6, 9.5, 9.9, 10.0, 0.05, 2.0],
                    [10.3, 8.9, 12.6, 12.5, 10.0, 0.05, 0.0],
                    [9.8, 11.0, 8.4, 7.0, 10.0, 0.05, 0.0],
                    [9.5, 9.8, 9.9, 9.9, 10.0, 0.05, 5.0],
                    [11.0, 10.0, 10.7, 10.1, 10.0, 0.05, 0.0],
                    [9.8, 10.7, 10.9, 9.6, 10.0, 0.05, 0.0],
                    [10.3, 11.5, 9.8, 10.9, 10.0, 0.05, 0.0],
                    [10.2, 10.0, 10.7, 9.9, 10.0, 0.05, 0.0]
                             ])
        np.testing.assert_array_almost_equal(hist.values, expected, decimal=2)

        hist2 = pd.DataFrame(data=[
            [0.95, 1.13, 0.98, 0.95, 100.0, 0.5, 0.0],
             [1.13, 1.09, 1.11, 0.86, 100.0, 0.5, 0.0],
             [1.0, 0.86, 0.95, 0.99, 100.0, 0.5, 4.0],
             [1.03, 0.89, 1.26, 1.25, 100.0, 0.5, 0.0],
             [0.98, 1.1, 0.84, 0.7, 100.0, 0.5, 0.0],
             [0.95, 0.98, 0.99, 0.99, 100.0, 0.5, 0.25],
             [1.1, 1.0, 1.07, 1.01, 100.0, 0.5, 0.0],
             [0.98, 1.07, 1.09, 0.96, 100.0, 0.5, 0.0],
             [1.03, 1.15, 0.98, 1.09, 100.0, 0.5, 0.0],
             [1.02, 1.0, 1.07, 0.99, 100.0, 0.5, 0.5],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ], columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
            index=pd.date_range(start="2021-01-01", periods=11, freq="D"),
        dtype=np.float64)
        expected = np.array([[0.475, 0.565, 0.49, 0.475, 200.0, 1.0, 0.0],
                             [0.565, 0.545, 0.555, 0.43, 200.0, 1.0, 0.0],
                             [0.5, 0.43, 0.475, 0.495, 200.0, 1.0, 4.0],
                             [0.515, 0.445, 0.63, 0.625, 200.0, 1.0, 0.0],
                             [0.49, 0.55, 0.42, 0.35, 200.0, 1.0, 0.0],
                             [0.475, 0.49, 0.495, 0.495, 200.0, 1.0, 0.25],
                             [0.55, 0.5, 0.535, 0.505, 200.0, 1.0, 0.0],
                             [0.49, 0.535, 0.545, 0.48, 200.0, 1.0, 0.0],
                             [0.515, 0.575, 0.49, 0.545, 200.0, 1.0, 0.0],
                             [0.51, 0.5, 0.535, 0.495, 200.0, 1.0, 0.5],
                             [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],])
        res = bcktst._reverse_split_norm(hist2)
        np.testing.assert_array_almost_equal(res.values, expected, decimal=2)

    def test_default_forge_last_candle(self):
        data = [
            {"AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                            name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
            "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                            name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY)},
            {"AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                            name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
            "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                            name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS)}
        ]
        bcktst = BackTest(data, MyStrat())
        actual = bcktst.default_forge_last_candle(data, 1,
                                                datetime.fromisoformat("2024-01-09 00:00:00"), 0)
        expected = [
            (181.85793785162858, 185.3634744030773, 181.26869341477493, 184.904052734375, 101_986_300, 0, "AAPL"),
            (495.1199951171875, 543.25, 494.7900085449219, 531.4000244140625, 141_561_000, 0, "NVDA")
        ]
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i][0], actual[i][0])
            self.assertAlmostEqual(expected[i][1], actual[i][1])
            self.assertAlmostEqual(expected[i][2], actual[i][2])
            self.assertAlmostEqual(expected[i][3], actual[i][3])
            self.assertEqual(expected[i][4], actual[i][4])
            self.assertEqual(expected[i][5], actual[i][5])

    def test_prepare_data(self):
        pd.options.mode.chained_assignment = None  # default='warn'
        data = [
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                            name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                            name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS),
                "TSLA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                            name="TSLA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS)
             },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                            name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                            name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
                "TSLA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                            name="TSLA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS)
            }
        ]
        # Add padding to simulate IPO and delisting
        data[0]["TSLA"].data.loc[:"2024-01-01"] = np.nan
        data[1]["TSLA"].data.loc[:"2024-01-01"] = np.nan
        data[0]["AAPL"].data.loc["2024-02-01":] = np.nan
        data[1]["AAPL"].data.loc["2024-02-01":] = np.nan
        data[0]["AAPL"].data["Stock Splits"].loc["2024-01-08 00:00:00"] = 5.0
        data[1]["AAPL"].data["Stock Splits"].loc["2024-01-08 00:00:00"] = 5.0
        data[0]["NVDA"].data["Stock Splits"].loc["2024-01-02 00:00:00"] = 0.25
        data[1]["NVDA"].data["Stock Splits"].loc["2023-12-29 00:00:00"] = 0.25

        bcktst = BackTest(data, MyStrat())
        for i in range(2):
            for ticker in data[i].keys():
                data[i][ticker].data = bcktst._reverse_split_norm(data[i][ticker].data)
        actual = bcktst._prepare_data(data, 0, datetime.fromisoformat("2024-01-09 00:00:00"), 3,
                                      False, False, 0.)
        aapl_expected = np.array([
            [181.75807721578167, 182.5270849545475, 179.94038931104083, 180.9490966796875, 62303300, 0.0, 0.0],
            [181.85793785162858, 185.3634744030773, 181.26869341477493, 185.32351684570312, 59144500, 0.0, 5.0],
            [183.68560631116785, 184.91403450398255, 182.4971204361262, 184.904052734375, 42841800, 0.0, 0.0]]
        )
        nvda_tsla_expected = np.array([
            [484.6199951171875, 495.4700012207031, 483.05999755859375, 490.9700012207031, 41456800, 0.0, 0.0],
            [495.1199951171875, 522.75, 494.7900085449219, 522.530029296875, 64251000, 0.0, 0.0],
            [524.010009765625, 543.25, 516.9000244140625, 531.4000244140625, 77310000, 0.0, 0.0]
        ])
        expected = [
            Record(pd.DataFrame(data=aapl_expected, columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                dtype=np.float64), "AAPL", 0, False, False,
                   DividendFrequency.QUARTERLY, short_rate=0.0,
                   next_tick=pd.Series([184.1150616539879,186.16243684463709,183.68560174206004,185.9527130126953,46792900,0.0],
                                       index=["Open", "High", "Low", "Close", "Volume", "Dividends"])),
            Record(pd.DataFrame(data=nvda_tsla_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64), "NVDA", 0, False, False,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.0,
                   next_tick=pd.Series(
                       [536.1599731445312,546.0,534.8900146484375,543.5,53379600,0.0],
                       index=["Open", "High", "Low", "Close", "Volume", "Dividends"])),
            Record(pd.DataFrame(data=nvda_tsla_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64), "TSLA", 0, False, False,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.0,
                   next_tick=pd.Series(
                       [536.1599731445312,546.0,534.8900146484375,543.5,53379600,0.0],
                       index=["Open", "High", "Low", "Close", "Volume", "Dividends"])),
        ]
        for i in range(3):
            np.testing.assert_array_almost_equal(actual[i].chart.values, expected[i].chart.values)
            self.assertEqual(actual[i].ticker, expected[i].ticker)
            self.assertEqual(actual[i].time_res, expected[i].time_res)
            self.assertEqual(actual[i].marginable, expected[i].marginable)
            self.assertEqual(actual[i].shortable, expected[i].shortable)
            self.assertEqual(actual[i].div_freq, expected[i].div_freq)
            self.assertEqual(actual[i].short_rate, expected[i].short_rate)
            np.testing.assert_array_almost_equal(actual[i].next_tick.values, expected[i].next_tick.values)

        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        # TODO: Make a new test to test padding, split for next day and custom short rate/shortable/marginable (IN ONE TEST)
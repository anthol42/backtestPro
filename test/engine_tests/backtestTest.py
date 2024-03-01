from unittest import TestCase
import pandas as pd
from src.backtest.engine import Position
import numpy as np
from datetime import datetime, timedelta
from src.backtest.engine import TSData
from src.backtest.engine import Strategy
from typing import List
from src.backtest.engine import Record
from src.backtest.engine import DividendFrequency
from src.backtest.engine import Backtest
from copy import deepcopy


class MyStrat(Strategy):
    def run(self, data: List[List[Record]], timestep: datetime):
        pass

class TestBacktest(TestCase):
    def test_reverse_split_norm(self):
        DATA = pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date")
        data = [
            {"AAPL": TSData(DATA, name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY)}
        ]
        bcktst = Backtest(data, MyStrat())
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
        bcktst = Backtest(data, MyStrat())
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
        data[0]["NVDA"].data["Stock Splits"].loc["2024-01-03 00:00:00"] = 0.25
        data[1]["NVDA"].data["Stock Splits"].loc["2024-01-03 00:00:00"] = 0.25

        bcktst = Backtest(data, MyStrat())
        # Reverse split normalization
        for i in range(2):
            for ticker in data[i].keys():
                data[i][ticker].data = bcktst._reverse_split_norm(data[i][ticker].data)
        actual = bcktst._prepare_data(data, 0, datetime.fromisoformat("2024-01-09 00:00:00"), 3,
                                      False, False, 0., save_next_tick=True)
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

        # Make AAPL delisted
        data[0]["AAPL"].data.loc["2024-01-01":] = np.nan
        data[1]["AAPL"].data.loc["2024-01-01":] = np.nan
        # TSLA is already IPO, so I should get a window of 1 day
        data[0]["NVDA"].data["Short_rate"] = 0.1
        data[1]["NVDA"].data["Short_rate"] = 0.1
        data[0]["NVDA"].data["Marginable"] = False
        data[1]["NVDA"].data["Marginable"] = False
        data[0]["NVDA"].data["Shortable"] = True
        data[1]["NVDA"].data["Shortable"] = True

        current_time = '2024-01-02 00:00:00'
        nvda_expected = np.array([
            [496.42999267578125, 498.8399963378906, 494.1199951171875, 495.2200012207031, 24658700, 0.0, 0.0],
            [498.1300048828125, 499.9700012207031, 487.510009765625, 495.2200012207031, 38869000, 0.0, 0.0],
            [492.44000244140625, 492.95001220703125, 475.95001220703125, 481.67999267578125, 41125400, 0.0, 0.0]
        ])
        nvda_expected[:, :-3] /= 4
        nvda_expected[:, -3] *= 4
        nvda_df = pd.DataFrame(data=nvda_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64)
        nvda_df["Short_rate"] = 0.1
        nvda_df["Marginable"] = False
        nvda_df["Shortable"] = True
        tsla_expected = np.array([
            [492.44000244140625, 492.95001220703125, 475.95001220703125, 481.67999267578125, 41125400, 0.0, 0.0]
        ])

        expected = [
            Record(None, "AAPL", 0, False, False,
                   DividendFrequency.QUARTERLY, short_rate=0.0,
                   next_tick=None,
                   ),
            Record(nvda_df, "NVDA", 0, False, True,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.1,
                   next_tick=pd.Series(
                       [474.8500061035156,481.8399963378906,473.20001220703125,475.69000244140625,32089600,0.0],
                       index=["Open", "High", "Low", "Close", "Volume", "Dividends"])),
            Record(pd.DataFrame(data=tsla_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64), "TSLA", 0, False, False,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.0,
                   next_tick=pd.Series(
                       [474.8500061035156,481.8399963378906,473.20001220703125,475.69000244140625,32089600,0.0],
                       index=["Open", "High", "Low", "Close", "Volume", "Dividends"])),
        ]
        actual = bcktst._prepare_data(data, 0, datetime.fromisoformat(current_time), 3,
                                      False, False, 0., save_next_tick=True)

        for i in range(3):
            if expected[i].chart is None:
                self.assertIsNone(actual[i].chart)
                self.assertEqual(actual[i].next_tick, expected[i].next_tick)    # None
            else:
                np.testing.assert_array_almost_equal(actual[i].chart.values, expected[i].chart.values)
                np.testing.assert_array_almost_equal(actual[i].next_tick.values, expected[i].next_tick.values)
            self.assertEqual(actual[i].ticker, expected[i].ticker)
            self.assertEqual(actual[i].time_res, expected[i].time_res)
            self.assertEqual(actual[i].marginable, expected[i].marginable)
            self.assertEqual(actual[i].shortable, expected[i].shortable)
            self.assertEqual(actual[i].div_freq, expected[i].div_freq)
            self.assertEqual(actual[i].short_rate, expected[i].short_rate)


        # Now test for another time resolution: 5d
        aapl_expected = np.array([
            [193.6528866387712, 194.15224943592054, 191.48565392786432, 192.28463745117188, 42628800, 0.0, 0.0]
        ])
        # Because of split that didn't happen yet:
        aapl_expected[:, :-3] *= 5
        aapl_expected[:, -3] /= 5
        nvda_expected = np.array([
            [498.1300048828125,499.9700012207031,487.510009765625,495.2200012207031,38869000,0.0,0.0]
        ])
        nvda_expected[:, :-3] /= 4
        nvda_expected[:, -3] *= 4
        nvda_df = pd.DataFrame(data=nvda_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64)
        nvda_df["Short_rate"] = 0.1
        nvda_df["Marginable"] = False
        nvda_df["Shortable"] = True
        expected = [
            Record(pd.DataFrame(data=aapl_expected,
                                columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                                dtype=np.float64), "AAPL", 1, False, False,
                   DividendFrequency.QUARTERLY, short_rate=0.0,
                   next_tick=None,
                   ),
            Record(nvda_df, "NVDA", 1, False, True,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.1,
                   next_tick=None),
            # The TSLA is not IPO yet for this time resolution
            Record(None, "TSLA", 1, False, False,
                   DividendFrequency.NO_DIVIDENDS, short_rate=0.0,
                   next_tick=None),
        ]
        max_look_back_dt = datetime.fromisoformat('2023-12-29 00:00:00')
        actual = bcktst._prepare_data(data, 1, max_look_back_dt, 3,
                                      False, False, 0.,
                                      max_look_back_dt=max_look_back_dt, save_next_tick=False)
        for i in range(3):
            if expected[i].chart is None:
                self.assertIsNone(actual[i].chart)
            else:
                np.testing.assert_array_almost_equal(actual[i].chart.values, expected[i].chart.values)
            self.assertIsNone(actual[i].next_tick)
            self.assertEqual(actual[i].ticker, expected[i].ticker)
            self.assertEqual(actual[i].time_res, expected[i].time_res)
            self.assertEqual(actual[i].marginable, expected[i].marginable)
            self.assertEqual(actual[i].shortable, expected[i].shortable)
            self.assertEqual(actual[i].div_freq, expected[i].div_freq)
            self.assertEqual(actual[i].short_rate, expected[i].short_rate)
        pd.options.mode.chained_assignment = 'warn'  # default='warn'

    def test_initialize_bcktst(self):
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
        data[0]["NVDA"].data["Stock Splits"].loc["2024-01-03 00:00:00"] = 0.25
        data[1]["NVDA"].data["Stock Splits"].loc["2024-01-03 00:00:00"] = 0.25

        bcktst = Backtest(deepcopy(data), MyStrat())

        features, tickers, timesteps_list = bcktst._initialize_bcktst()
        expected_features = [
            "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
        ]
        expected_tickers = ["AAPL", "NVDA", "TSLA"]
        expected_timesteps = data[0]["NVDA"].data.index.tolist()
        self.assertEqual(expected_features, features)
        self.assertEqual(expected_tickers, tickers)
        self.assertEqual(expected_timesteps, timesteps_list)
        self.assertEqual([timedelta(days=1), timedelta(days=5)], bcktst.available_time_res)

        # Now, prepare normalized data
        data[0]["AAPL"].data.iloc[:, :-3] *= 5
        data[0]["AAPL"].data.iloc[:, -3:-1] /= 5
        data[1]["AAPL"].data.iloc[:, :-3] *= 5
        data[1]["AAPL"].data.iloc[:, -3:-1] /= 5
        data[0]["NVDA"].data.iloc[:, :-3] /= 4
        data[0]["NVDA"].data.iloc[:, -3:] *= 4
        data[1]["NVDA"].data.iloc[:, :-3] /= 4
        data[1]["NVDA"].data.iloc[:, -3:] *= 4
        # Verify that data is normalized correctly
        for ticker in data[0].keys():
            data[0][ticker].data = bcktst._reverse_split_norm(data[0][ticker].data)
            data[1][ticker].data = bcktst._reverse_split_norm(data[1][ticker].data)


        pd.options.mode.chained_assignment = 'warn'  # default='warn'

    def test_prep_data(self):
        data = [{
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"),
                               name="AAPL", div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"),
                               name="NVDA", div_freq=DividendFrequency.NO_DIVIDENDS),
            },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS),
             },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
            }
        ]
        bcktst = Backtest(deepcopy(data), MyStrat(), main_timestep=1, window=3, verbose=1)
        bcktst._initialize_bcktst()
        actual = bcktst._prep_data(datetime.fromisoformat('2024-01-09 00:00:00'))

        self.assertEqual(3, len(actual))
        for i in range(3):
            self.assertEqual(2, len(actual[i]))

        # Check if the data has the good index (No need to check normalization since it is done in _prepare_data)
        idx1h = [
            "2024-01-04 15:30:00",
            "2024-01-05 09:30:00",
            "2024-01-05 10:30:00",
            "2024-01-05 11:30:00",
            "2024-01-05 12:30:00",
            "2024-01-05 13:30:00",
            "2024-01-05 14:30:00",
            "2024-01-05 15:30:00",
            "2024-01-08 09:30:00",
            "2024-01-08 10:30:00",
            "2024-01-08 11:30:00",
            "2024-01-08 12:30:00",
            "2024-01-08 13:30:00",
            "2024-01-08 14:30:00",
            "2024-01-08 15:30:00",
            "2024-01-09 09:30:00",
            "2024-01-09 10:30:00",
            "2024-01-09 11:30:00",
            "2024-01-09 12:30:00",
            "2024-01-09 13:30:00",
            "2024-01-09 14:30:00",
            "2024-01-09 15:30:00"
        ]
        expected_1h_idx = pd.DatetimeIndex([datetime.fromisoformat(e) for e in idx1h])
        expected_1d_idx = pd.DatetimeIndex([
            "2024-01-05",
            "2024-01-08",
            "2024-01-09"
        ])
        expected_5d_idx = pd.DatetimeIndex([
            "2024-01-03 00:00:00",
            "2024-01-08 00:00:00"
        ])
        for ticker in data[0]:
            idx = [record.ticker for record in actual[0]].index(ticker)
            np.testing.assert_array_equal(expected_1h_idx, actual[0][idx].chart.index)
            np.testing.assert_array_equal(expected_1d_idx, actual[1][idx].chart.index)
            np.testing.assert_array_equal(expected_5d_idx, actual[2][idx].chart.index)

        # Now verify the last candle
        expected_aapl = np.array([181.85793785162858,185.3634744030773,181.26869341477493,184.904052734375,101_986_300,0.0,0.0])
        expected_nvda = np.array([495.1199951171875,543.25,494.7900085449219,531.4000244140625,141_561_000,0.0,0.0])
        np.testing.assert_array_almost_equal(expected_aapl, actual[2][0].chart.iloc[-1].values)
        np.testing.assert_array_almost_equal(expected_nvda, actual[2][1].chart.iloc[-1].values)


    def test_get_mask(self):
        data = [{
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"),
                               name="AAPL", div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"),
                               name="NVDA", div_freq=DividendFrequency.NO_DIVIDENDS),
            },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS),
             },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
            }
        ]
        bcktst = Backtest(deepcopy(data), MyStrat(), main_timestep=1, window=3, verbose=1)
        bcktst._initialize_bcktst()
        main_ts_data = bcktst._prep_data(datetime.fromisoformat('2024-01-09 00:00:00'))[1]
        main_ts_data[0].chart = None
        actual_mask = bcktst._get_mask(main_ts_data)
        expected_mask = np.array([False, True])
        np.testing.assert_array_equal(expected_mask, actual_mask)

    def test_prep_brokers_data(self):
        data = [{
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"),
                               name="AAPL", div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"),
                               name="NVDA", div_freq=DividendFrequency.NO_DIVIDENDS),
            },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS),
             },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
            }
        ]
        pd.options.mode.chained_assignment = None  # default='warn'
        data[1]["AAPL"].data["Marginable"] = True
        data[1]["NVDA"].data["Short_rate"] = 0.1
        data[1]["AAPL"].data["Marginable"].loc[:"2024-01-08"] = False
        data[1]["NVDA"].data["Short_rate"].loc[:"2024-01-08"] = 0.0
        pd.options.mode.chained_assignment = 'warn'

        bcktst = Backtest(deepcopy(data), MyStrat(), main_timestep=1, window=3, verbose=1, default_short_rate=0.)
        bcktst._initialize_bcktst()
        prepared_data = bcktst._prep_data(datetime.fromisoformat('2024-01-09 00:00:00'))[1]
        current_data, next_tick_data, marginables, dividends, div_freq, short_rate, security_names = bcktst._prep_brokers_data(prepared_data)

        expected_current_data = np.array([
            [183.68560631116785,184.91403450398255,182.4971204361262,184.904052734375],
            [524.010009765625,543.25,516.9000244140625,531.4000244140625]
        ])
        np.testing.assert_array_almost_equal(expected_current_data, current_data, decimal=5)
        expected_next_tick_data = np.array([
            [184.1150616539879,186.16243684463709,183.68560174206004,185.9527130126953],
            [536.1599731445312,546.0,534.8900146484375,543.5]
        ])
        np.testing.assert_array_almost_equal(expected_next_tick_data, next_tick_data, decimal=5)
        marginables_expected = np.array([[True, False],
                                         [False, False]])
        np.testing.assert_array_equal(marginables_expected, marginables)
        dividends_expected = np.zeros((2,), dtype=np.float32)
        np.testing.assert_array_equal(dividends_expected, dividends)

        expected_div_freq = [DividendFrequency.QUARTERLY, DividendFrequency.NO_DIVIDENDS]
        self.assertEqual(expected_div_freq, div_freq)

        expected_short_rate = np.array([0.0, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(expected_short_rate, short_rate)

        self.assertEqual(["AAPL", "NVDA"], security_names)

    """
    Note:
        The step and run method are not tested here and will be tested during the integration tests
    """

    def test_sell_all(self):
        data = [{
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime"),
                               name="AAPL", div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime"),
                               name="NVDA", div_freq=DividendFrequency.NO_DIVIDENDS),
            },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=1), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=1), div_freq=DividendFrequency.NO_DIVIDENDS),
             },
            {
                "AAPL": TSData(pd.read_csv("test_data/AAPL_1y_5d.csv", index_col="Date"),
                               name="AAPL", time_res=timedelta(days=5), div_freq=DividendFrequency.QUARTERLY),
                "NVDA": TSData(pd.read_csv("test_data/NVDA_1y_5d.csv", index_col="Date"),
                               name="NVDA", time_res=timedelta(days=5), div_freq=DividendFrequency.NO_DIVIDENDS),
            }
        ]
        bcktst = Backtest(deepcopy(data), MyStrat(), main_timestep=1, window=3, verbose=1)
        bcktst._initialize_bcktst()
        bcktst.broker._debt_record["NVDA"] = 0
        bcktst.broker.portfolio._long = {
            "NVDA": Position("NVDA", 100, True, 400,
                             datetime(2023, 12, 15, 9, 30), 1.)
        }
        bcktst.broker.portfolio._short = {
            "AAPL": Position("AAPL", 100, False, 105.25,
                             datetime(2023, 12, 10, 12, 15), 0.)
        }

        bcktst._sell_all(datetime.fromisoformat('2024-01-09 00:00:00'))
        self.assertEqual(0, bcktst.broker.portfolio._long["NVDA"].amount)
        self.assertEqual(0, bcktst.broker.portfolio._short["AAPL"].amount)


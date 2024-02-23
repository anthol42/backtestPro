from typing import List, Dict, Final, Tuple
from unittest import TestCase
import pandas as pd
from backtest.engine import TSData
import numpy as np
from datetime import datetime, timedelta
from backtest.engine.time_resolution_extenders import TimeResExtender, BasicExtender
from copy import deepcopy

class TestTimeResExtender(TestCase):
    def test_init(self):
        class ResExtender(TimeResExtender):
            n_out = 2
            out_res = [timedelta(days=3), timedelta(weeks=1)]
            def single_extend(self, data: TSData) -> Tuple[TSData]:
                pass

        class MonthExtender(TimeResExtender):
            n_out = 1
            out_res = [timedelta(days=30)]
            def single_extend(self, data: TSData) -> Tuple[TSData]:
                pass

        res_extender = ResExtender()
        month_extender = MonthExtender()
        self.assertEqual(2, res_extender.n_out)
        self.assertEqual([timedelta(days=3), timedelta(weeks=1)], res_extender.out_res)
        self.assertEqual(1, month_extender.n_out)
        self.assertEqual([timedelta(days=30)], month_extender.out_res)
        total_extender = res_extender + month_extender
        self.assertEqual(3, total_extender.n_out)
        self.assertEqual([timedelta(days=3), timedelta(weeks=1), timedelta(days=30)], total_extender.out_res)

    def test_add(self):
        def test_init(self):
            class ResExtender(TimeResExtender):
                n_out = 2
                out_res = [timedelta(days=3), timedelta(weeks=1)]

                def single_extend(self, data: TSData) -> Tuple[TSData]:
                    pass

            class MonthExtender(TimeResExtender):
                n_out = 1
                out_res = [timedelta(days=30)]

                def single_extend(self, data: TSData) -> Tuple[TSData]:
                    pass

            total_extender = ResExtender() + MonthExtender()
            self.assertEqual(3, total_extender.n_out)
            self.assertEqual([timedelta(days=3), timedelta(weeks=1), timedelta(days=30)], total_extender.out_res)
    def test_extend(self):
        class ResExtender(TimeResExtender):
            n_out = 2
            out_res = [timedelta(days=3), timedelta(weeks=1)]
            def single_extend(self, data: TSData) -> Tuple[TSData, ...]:
                pd.set_option('mode.chained_assignment', None)
                data3d = deepcopy(data)
                data1w = deepcopy(data)
                # 3 days
                ohlc_resampled = data.data.resample('3D').ohlc()

                # resample other columns such as volume and dividends
                if "Stock Splits" in data.data.columns:
                    data.data.loc[data.data["Stock Splits"] == 0, "Stock Splits"] = 1
                    stock_splits_resampled = data.data["Stock Splits"].resample('3D').prod()
                    stock_splits_resampled.loc[stock_splits_resampled == 1] = 0
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close", "Stock Splits"]).resample('3D').sum()
                    other_resampled["Stock Splits"] = stock_splits_resampled
                else:
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close"]).resample('3D').sum()

                # combine OHLC and other columns into one DataFrame
                data3d.data = pd.concat([ohlc_resampled, other_resampled], axis=1)
                data3d.time_res = timedelta(days=3)

                # 1 week
                ohlc_resampled = data.data.resample('1W').ohlc()
                # resample other columns such as volume and dividends
                if "Stock Splits" in data.data.columns:
                    data.data.loc[data.data["Stock Splits"] == 0, "Stock Splits"] = 1
                    stock_splits_resampled = data.data["Stock Splits"].resample('1W').prod()
                    stock_splits_resampled.loc[stock_splits_resampled == 1] = 0
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close", "Stock Splits"]).resample(
                        '1W').sum()
                    other_resampled["Stock Splits"] = stock_splits_resampled
                else:
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close"]).resample('1W').sum()

                # combine OHLC and other columns into one DataFrame
                data1w.data = pd.concat([ohlc_resampled, other_resampled], axis=1)
                data1w.time_res = timedelta(weeks=1)
                pd.set_option('mode.chained_assignment', 'warn')
                return data3d, data1w

        class MonthExtender(TimeResExtender):
            n_out = 1
            out_res = [timedelta(days=30)]
            def single_extend(self, data: TSData) -> Tuple[TSData]:
                pd.set_option('mode.chained_assignment', None)
                data1m = deepcopy(data)
                ohlc_resampled = data.data.resample('30D').ohlc()

                # resample other columns such as volume and dividends
                if "Stock Splits" in data.data.columns:
                    data.data.loc[data.data["Stock Splits"] == 0, "Stock Splits"] = 1
                    stock_splits_resampled = data.data["Stock Splits"].resample('3D').prod()
                    stock_splits_resampled.loc[stock_splits_resampled == 1] = 0
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close", "Stock Splits"]).resample(
                        '30D').sum()
                    other_resampled["Stock Splits"] = stock_splits_resampled
                else:
                    other_resampled = data.data.drop(columns=["Open", "High", "Low", "Close"]).resample('30D').sum()

                # combine OHLC and other columns into one DataFrame
                data1m.data = pd.concat([ohlc_resampled, other_resampled], axis=1)
                data1m.time_res = timedelta(days=30)
                pd.set_option('mode.chained_assignment', 'warn')
                return data1m,

        aapl = pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date", parse_dates=True)
        nvda = pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date", parse_dates=True)

        data = [
            {
                "AAPL": TSData(aapl, name="AAPL_6mo_1d"),
                "NVDA": TSData(nvda, name="NVDA_6mo_1d")
            }
        ]

        extender = ResExtender() + MonthExtender()
        new_data = data + extender.extend(data)
        self.assertEqual(4, len(new_data))
        self.assertEqual(timedelta(days=1), new_data[0]["AAPL"].time_res)
        self.assertEqual(timedelta(days=3), new_data[1]["AAPL"].time_res)
        self.assertEqual(timedelta(weeks=1), new_data[2]["AAPL"].time_res)
        self.assertEqual(timedelta(days=30), new_data[3]["AAPL"].time_res)
        self.assertEqual(timedelta(days=1), new_data[0]["NVDA"].time_res)
        self.assertEqual(timedelta(days=3), new_data[1]["NVDA"].time_res)
        self.assertEqual(timedelta(weeks=1), new_data[2]["NVDA"].time_res)
        self.assertEqual(timedelta(days=30), new_data[3]["NVDA"].time_res)

class TestBasicExtender(TestCase):
    def test_7d(self):
        aapl = pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date", parse_dates=True)
        nvda = pd.read_csv("test_data/NVDA_6mo_1d.csv", index_col="Date", parse_dates=True)

        data = [
            {
                "AAPL": TSData(aapl, name="AAPL_6mo_1d"),
                "NVDA": TSData(nvda, name="NVDA_6mo_1d")
            }
        ]

        # Verify start and end points
        extender = BasicExtender("1w")
        new_data = data + extender.extend(data)
        aapl_start_expected = np.array([176.60169734566966,181.08007898396926,175.36491809973708,178.14768981933594,201_202_400,0.0,0.0])
        aapl_end_expected = np.array([181.7899932861328,184.9499969482422,180.0,184.47999572753906,127_687_122,0.0,0.0])
        np.testing.assert_array_almost_equal(aapl_start_expected, new_data[-1]["AAPL"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(aapl_end_expected, new_data[-1]["AAPL"].data.iloc[-1].values, decimal=5)

        nvda_start_expected = np.array([481.2680145912384,502.57439674585044,450.163313886704,460.10162353515625,361_763_800,0.0,0.0])
        nvda_end_expected = np.array([719.469970703125,781.5399780273438, 662.47998046875,778.8800048828125,209_282_961,0.0,0.0])
        np.testing.assert_array_almost_equal(nvda_start_expected, new_data[-1]["NVDA"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(nvda_end_expected, new_data[-1]["NVDA"].data.iloc[-1].values, decimal=5)

    def test_4h(self):
        aapl = pd.read_csv("test_data/AAPL_6mo_1h.csv", index_col="Datetime", parse_dates=True)
        nvda = pd.read_csv("test_data/NVDA_6mo_1h.csv", index_col="Datetime", parse_dates=True)

        data = [
            {
                "AAPL": TSData(aapl, name="AAPL_6mo_1h"),
                "NVDA": TSData(nvda, name="NVDA_6mo_1h")
            }
        ]

        extender = BasicExtender("4h")
        new_data = data + extender.extend(data)
        aapl_start_expected = np.array([178.520004, 181.331100, 178.324997, 180.729996, 12913906, 0.0, 0.0])
        aapl_end_expected = np.array([183.070007, 184.955002, 183.000000, 184.369995, 22612349, 0.0, 0.0])
        np.testing.assert_array_almost_equal(aapl_start_expected, new_data[-1]["AAPL"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(aapl_end_expected, new_data[-1]["AAPL"].data.iloc[-1].values, decimal=5)

        nvda_start_expected = np.array([460.478394, 469.079987, 452.079987, 466.968994, 14339331,0.0,0.0])
        nvda_end_expected = np.array([774.770020, 785.750000, 770.549988, 785.270020, 24348236,0.0,0.0])
        np.testing.assert_array_almost_equal(nvda_start_expected, new_data[-1]["NVDA"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(nvda_end_expected, new_data[-1]["NVDA"].data.iloc[-1].values, decimal=5)

        # Test that the static time resolution didn't changed (because of modifications by reference)
        aapl_start_expected = np.array([178.52000427246094,180.38999938964844,178.3249969482422,180.10499572753906,0,0.0,0.0])
        aapl_end_expected = np.array([184.56739807128906,184.6199951171875,184.17999267578125,184.3699951171875,7964831,0.0,0.0])
        np.testing.assert_array_almost_equal(aapl_start_expected, new_data[0]["AAPL"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(aapl_end_expected, new_data[0]["AAPL"].data.iloc[-1].values, decimal=5)

        nvda_start_expected = np.array([460.4783935546875,466.0,452.0799865722656,463.3599853515625,0,0.0,0.0])
        nvda_end_expected = np.array([781.7150268554688,785.75,780.2899780273438,785.27001953125,6582620,0.0,0.0])
        np.testing.assert_array_almost_equal(nvda_start_expected, new_data[0]["NVDA"].data.iloc[0].values, decimal=5)
        np.testing.assert_array_almost_equal(nvda_end_expected, new_data[0]["NVDA"].data.iloc[-1].values, decimal=5)



from unittest import TestCase
from src.backtest.indicators.indicator import Indicator
import numpy as np
import pandas as pd


class TestIndicator(TestCase):
    def test_set_cb(self):
        def MyCB(data, index, features, previous_data, period):
            return data
        ind = Indicator(["None"])
        ind.set_callback(MyCB)
        self.assertEqual(ind._cb, MyCB)

    def test_set_params(self):
        ind = Indicator(["None"], period=int)
        ind.set_params(period=14)
        self.assertEqual({"period": 14}, ind.params)

    def test_call(self):
        def MyCB(data, index, features, previous_data, period):
            return data
        ind = Indicator(["None"], period=int)
        ind = ind(MyCB)
        self.assertEqual(MyCB, ind._cb)
        ind = ind(period=14)
        self.assertEqual({"period": 14}, ind.params)

    def test_get(self):
        data = pd.DataFrame(np.random.rand(10, 4), index=pd.date_range("2020-01-01", periods=10), columns=["Open", "High", "Low", "Close"])
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return out[:, np.newaxis]

        ind = SMA(period=2)
        out = ind.get(data, None)
        self.assertEqual({"period": 2}, ind.params)
        self.assertEqual(10, len(out))
        self.assertEqual(["SMA"], out.columns.tolist())

        # Now, test with previous data
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return previous_data

        ind = SMA(period=4)
        prev_data = pd.DataFrame(np.random.rand(10, 1), index=pd.date_range("2020-01-01", periods=10), columns=["SMA"])
        out = ind.get(data, prev_data)
        np.testing.assert_array_equal(prev_data.to_numpy(), out.to_numpy())

    def test_out(self):
        ind = Indicator(["SMA"], period=int)
        self.assertEqual(["SMA"], ind.out)
        ind = Indicator(["SMA", "EMA"], period=int)
        self.assertEqual(["SMA", "EMA"], ind.out)
        ind.set_id(1)
        self.assertEqual(["SMA_1", "EMA_1"], ind.out)




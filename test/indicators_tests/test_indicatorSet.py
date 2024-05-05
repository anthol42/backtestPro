from unittest import TestCase
from src.backtest.indicators.indicator import Indicator
from src.backtest.indicators.indicatorSet import IndicatorSet
import numpy as np
import pandas as pd

class TestIndicatorSet(TestCase):
    def test_init(self):
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return out[:, np.newaxis]

        @Indicator(["EMA", "Signal"], period=int)
        def EMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            out[0] = data[0, 0]
            alpha = 2 / (period + 1)
            for i in range(1, len(data)):
                out[i] = alpha * data[i, 0] + (1 - alpha) * out[i - 1]
            return np.stack([out, out], axis=1)

        ind_set = IndicatorSet(SMA(period=2),
                               EMA(period=3)
                               )
        self.assertEqual(2, len(ind_set))
        self.assertEqual(['SMA(2)', 'EMA(3)'], [ind.name for ind in ind_set._indicators])
        self.assertEqual(['SMA(2)', 'EMA(3)', 'Signal(3)'], ind_set.out)

        # Now, test with duplicates
        ind_set = IndicatorSet(EMA(period=2),
                               SMA(period=3),
                               EMA(period=3)
                               )
        self.assertEqual(3, len(ind_set))
        self.assertEqual(['EMA(2)', 'SMA(3)', 'EMA(3)'], [ind.name for ind in ind_set._indicators])
        self.assertEqual(['EMA(2)', 'Signal(2)', 'SMA(3)', 'EMA(3)', 'Signal(3)'], ind_set.out)

    def test_add(self):
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return out[:, np.newaxis]

        @Indicator(["EMA", "Signal"], period=int)
        def EMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            out[0] = data[0, 0]
            alpha = 2 / (period + 1)
            for i in range(1, len(data)):
                out[i] = alpha * data[i, 0] + (1 - alpha) * out[i - 1]
            return np.stack([out, out], axis=1)

        ind_set = IndicatorSet(SMA(period=2))
        ind_set.add(EMA(period=2))
        self.assertEqual(2, len(ind_set))
        self.assertEqual(["SMA(2)", "EMA(2)"], [ind.name for ind in ind_set._indicators])
        self.assertEqual(["SMA(2)", "EMA(2)", "Signal(2)"], ind_set.out)

        # Now, test with duplicates
        ind_set.add(EMA(period=3))
        self.assertEqual(3, len(ind_set))
        self.assertEqual(['SMA(2)', 'EMA(2)', 'EMA(3)'], [ind.name for ind in ind_set._indicators])
        self.assertEqual(['SMA(2)', 'EMA(2)', 'Signal(2)', 'EMA(3)', 'Signal(3)'], ind_set.out)

    def test_run_all(self):
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return out[:, np.newaxis]

        @Indicator(["EMA", "Signal"], period=int)
        def EMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            out[0] = data[0, 0]
            alpha = 2 / (period + 1)
            for i in range(1, len(data)):
                out[i] = alpha * data[i, 0] + (1 - alpha) * out[i - 1]
            return np.stack([out, out], axis=1)

        data = pd.DataFrame(np.random.rand(10, 4), index=pd.date_range("2020-01-01", periods=10), columns=["Open", "High", "Low", "Close"])
        ind_set = IndicatorSet(SMA(period=2),
                               EMA(period=3)
                               )
        out = ind_set.run_all(data, None)
        self.assertEqual(10, len(out))
        self.assertEqual(['Open', 'High', 'Low', 'Close', 'SMA(2)', 'EMA(3)', 'Signal(3)'], out.columns.tolist())

        # Now, test with previous data
        prev_data = pd.DataFrame(np.random.rand(10, 3), index=pd.date_range("2020-01-01", periods=10), columns=['SMA(4)', 'EMA(3)', 'Signal(3)'])
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return previous_data[:, -1][:, np.newaxis]

        ind_set = IndicatorSet(SMA(period=4),
                               EMA(period=3)
                               )
        out = ind_set.run_all(data, prev_data)
        np.testing.assert_array_equal(prev_data["SMA(4)"], out["SMA(4)"])

        # Now, test with multiple indicators with the same name
        @Indicator(["SMA"], period=int)
        def SMA(data: np.ndarray, index: list, features: list, previous_data: np.ndarray, period: int = 3) -> np.ndarray:
            out = np.zeros(len(data), dtype=np.float32)
            for i in range(len(data) - period + 1):
                out[i] = data[i: i+period, 0].sum() / period
            return out[:, np.newaxis]
        ind_set = IndicatorSet(SMA(period=2),
                               EMA(period=3),
                               SMA(period=3)
                               )
        out = ind_set.run_all(data, None)
        self.assertEqual(10, len(out))
        self.assertEqual(['Open', 'High', 'Low', 'Close', 'SMA(2)', 'EMA(3)', 'Signal(3)', 'SMA(3)'], out.columns.tolist())

from unittest import TestCase
from backtest.src.tsData import TSData, DividendFrequency
import pandas as pd
from datetime import timedelta

class TestDividendFrequency(TestCase):
    def test_from_delta(self):
        self.assertEqual(DividendFrequency.from_delta(30), DividendFrequency.MONTHLY)
        self.assertEqual(DividendFrequency.from_delta(90), DividendFrequency.QUARTERLY)
        self.assertEqual(DividendFrequency.from_delta(180), DividendFrequency.BIANNUALLY)
        self.assertEqual(DividendFrequency.from_delta(365), DividendFrequency.YEARLY)

        # Not equal
        self.assertEqual(DividendFrequency.from_delta(28), DividendFrequency.MONTHLY)
        self.assertEqual(DividendFrequency.from_delta(33), DividendFrequency.MONTHLY)
        self.assertEqual(DividendFrequency.from_delta(88), DividendFrequency.QUARTERLY)
        self.assertEqual(DividendFrequency.from_delta(92), DividendFrequency.QUARTERLY)
        self.assertEqual(DividendFrequency.from_delta(175), DividendFrequency.BIANNUALLY)
        self.assertEqual(DividendFrequency.from_delta(185), DividendFrequency.BIANNUALLY)
        self.assertEqual(DividendFrequency.from_delta(360), DividendFrequency.YEARLY)
        self.assertEqual(DividendFrequency.from_delta(370), DividendFrequency.YEARLY)



class TestTSData(TestCase):
    # Load test data
    data1d = pd.read_csv("test_data/AAPL_6mo_1d.csv", index_col="Date", parse_dates=True)
    data5d = pd.read_csv("./test_data/AAPL_1y_5d.csv", index_col="Date", parse_dates=True)

    def test_init(self):
        # Test init with 1d data
        tsData1d = TSData(self.data1d, name="AAPL_3mo_1d")
        self.assertEqual(tsData1d.name, "AAPL_3mo_1d")
        self.assertEqual(tsData1d.time_res, timedelta(days=1))
        self.assertEqual(tsData1d.div_freq, DividendFrequency.QUARTERLY)

        # Test init with 5d data
        tsData5d = TSData(self.data5d, name="AAPL_1y_5d")
        self.assertEqual(tsData5d.name, "AAPL_1y_5d")
        self.assertEqual(tsData5d.time_res, timedelta(days=5))
        self.assertEqual(tsData5d.div_freq, DividendFrequency.QUARTERLY)

        # Test init without detecting freq and time res
        tsData1w = TSData(self.data5d, name="AAPL_1y_5d", time_res=timedelta(weeks=1), div_freq=DividendFrequency.Quarterly)
        self.assertEqual(tsData1w.name, "AAPL_1y_5d")
        self.assertEqual(tsData1w.time_res, timedelta(weeks=1))
        self.assertEqual(tsData1w.div_freq, DividendFrequency.QUARTERLY)

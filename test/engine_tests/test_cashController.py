from unittest import TestCase
from src.backtest.engine import CashControllerBase, SimpleCashController, Account, Broker, Strategy
from src.backtest.engine.cashController import CashControllerTimeframe
import pandas as pd
from datetime import timedelta

class StudStrat(Strategy):
    def run(self, data: pd.DataFrame, timestep: pd.Timestamp):
        pass
class TestCashController(TestCase):
    def test_overall(self):
        cc = SimpleCashController(every_week=10, every_month=-50)
        cc.init(Account(), Broker(Account()), StudStrat())


        timestamps = pd.date_range("2020-01-01", "2020-03-01", freq="D")    # 2 months
        last_timestamp = timestamps[0]
        for ts in timestamps:
            if ts.week != last_timestamp.week:
                print("Week", ts)
                cc.deposit(ts, CashControllerTimeframe.WEEK)
            elif ts.month != last_timestamp.month:
                print("Month", ts)
                cc.deposit(ts, CashControllerTimeframe.MONTH)
            last_timestamp = ts

        self.assertEqual(cc.total_deposited, 10*8 - 2*50)
        self.assertEqual(cc.monthly_variation(ts), -50*2 + 10*4)

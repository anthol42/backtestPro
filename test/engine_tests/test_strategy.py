from unittest import TestCase
from src.backtest.engine.strategy import Strategy, Broker, Account
import os
from pathlib import PurePath
from datetime import timedelta

class MyStrat(Strategy):
    def __init__(self, prev_data=None):
        super().__init__()
        self.prev_data = prev_data
        self.i = 0
    def run(self, data, timestep):
        pass


class TestStrategy(TestCase):
    def test_save_load(self):
        strat = MyStrat({"Hello": "World"})
        if not os.path.exists(".cache"):
            os.mkdir(".cache")

        strat.save(PurePath(".cache/strat.pkl"))

        loaded_strat = Strategy.load(PurePath(".cache/strat.pkl"))

        self.assertEqual(loaded_strat.prev_data, {"Hello": "World"})
        self.assertEqual(loaded_strat.i, 0)

        strat.i = 42
        strat.prev_data = {"World": "Hello"}
        strat.save(PurePath(".cache/strat.pkl"))

        loaded_strat = Strategy.load(PurePath(".cache/strat.pkl"))
        self.assertEqual(loaded_strat.prev_data, {"World": "Hello"})
        self.assertEqual(loaded_strat.i, 42)


        # Now, test with a a broker, an account an available time resolutions
        strat = MyStrat({"Hello": "World"})
        account = Account()
        broker = Broker(account)
        strat.init(account, broker, [timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=15)])
        self.assertEqual(strat.account, account)
        self.assertEqual(strat.broker, broker)
        self.assertEqual(strat.available_time_res, [timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=15)])

        strat.save(PurePath(".cache/strat.pkl"))

        loaded_strat = Strategy.load(PurePath(".cache/strat.pkl"))
        self.assertEqual(loaded_strat.prev_data, {"Hello": "World"})
        self.assertEqual(loaded_strat.i, 0)
        self.assertIsNone(loaded_strat.account)
        self.assertIsNone(loaded_strat.broker)
        self.assertIsNone(loaded_strat.available_time_res)
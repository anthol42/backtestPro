import os
from unittest import TestCase

from src.backtest import RecordsBucket
from src.backtest.serve.renderers.log_renderer import LogRenderer
from src.backtest.data import Fetch, ToTSData
from src.backtest.serve.state_signals import StateSignals
from src.backtest.engine import Portfolio, Account, Broker, TradeOrder, TradeType, Strategy, SimpleCashController
from datetime import datetime
import pandas as pd
from pathlib import PurePath

orders = {
    "AAPL": TradeOrder(datetime(2024, 1, 12), "AAPL", (None, None), 100,
                       100, TradeType.BuyLong, None),
    "MSFT": TradeOrder(datetime(2024, 1, 12), "MSFT", (None, None), 100,
                          100, TradeType.BuyLong, None),
    "GOOG": TradeOrder(datetime(2024, 1, 12), "GOOG", (None, None), 100,
                            100, TradeType.SellLong, None),
    "AMZN": TradeOrder(datetime(2024, 1, 12), "AMZN", (None, None), 100,
                            100, TradeType.SellLong, None),
}

class MyStrat(Strategy):
    def run(self, data: RecordsBucket, timestep: datetime):
        pass

@Fetch
def IndexPipe(frm: datetime, to: datetime, *args, **kwargs):
    chart1 = pd.read_csv("../engine_tests/test_data/SPY_6mo_1d.csv", index_col="Date")
    chart1.index = pd.DatetimeIndex(["-".join(str(x).split("-")[:-1]) for x in chart1.index])
    return {"SPY": chart1}

class TestLogRenderer(TestCase):
    def test_render(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        account = Account(1000)
        broker = Broker(account)
        idx_pipe = IndexPipe() | ToTSData()
        state = StateSignals(account, broker, orders, MyStrat(), datetime(2024, 1, 12),
                             SimpleCashController(), 10, idx_pipe.get(None, None))

        # Now, try storing only the signals
        renderer = LogRenderer()
        renderer.render(state, PurePath(".cache"))

        # Check that the signals are stored, no need to check the format because each traderOrder handles that
        with open(PurePath(".cache") / "signals" / "signals.log", "r") as f:
            data = f.readlines()
            self.assertEqual(data[0], "timestamp,security,signal_type,price_lower_limit,price_upper_limit,n_shares,n_shares_borrowed,expiry\n")
            self.assertEqual(data[1], "2024-01-12T00:00:00,AAPL,BuyLong,None,None,100,100,None\n")
            self.assertEqual(data[2], "2024-01-12T00:00:00,MSFT,BuyLong,None,None,100,100,None\n")
            self.assertEqual(data[3], "2024-01-12T00:00:00,GOOG,SellLong,None,None,100,100,None\n")
            self.assertEqual(data[4], "2024-01-12T00:00:00,AMZN,SellLong,None,None,100,100,None\n")

        # Now, try storing everything and check if they are stored
        renderer = LogRenderer()
        renderer.render(state, PurePath(".cache"))

        with open(PurePath(".cache") / "signals" / "signals.log", "r") as f:
            data = f.readlines()
            self.assertEqual(data[0], "timestamp,security,signal_type,price_lower_limit,price_upper_limit,n_shares,n_shares_borrowed,expiry\n")
            self.assertEqual(data[1], "2024-01-12T00:00:00,AAPL,BuyLong,None,None,100,100,None\n")
            self.assertEqual(data[2], "2024-01-12T00:00:00,MSFT,BuyLong,None,None,100,100,None\n")
            self.assertEqual(data[3], "2024-01-12T00:00:00,GOOG,SellLong,None,None,100,100,None\n")
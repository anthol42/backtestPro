from unittest import TestCase

from src.backtest import RecordsBucket
from src.backtest.serve.renderers.json_renderer import JSONRenderer
from src.backtest.data import Fetch, ToTSData
from src.backtest.serve.state_signals import StateSignals
from src.backtest.engine import Portfolio, Account, Broker, TradeOrder, TradeType, Strategy
from datetime import datetime
import pandas as pd
from pathlib import PurePath
import json


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
class TestJSONRenderer(TestCase):
    def test_render(self):
        account = Account(1000)
        broker = Broker(account)
        idx_pipe = IndexPipe() | ToTSData()
        state = StateSignals(account, broker, orders, MyStrat(), datetime(2024, 1, 12), idx_pipe.get(None, None))

        # Now, try storing only the signals
        renderer = JSONRenderer()
        renderer.render(state, PurePath(".cache"))

        # Check that the signals are stored, no need to check the format because each traderOrder handles that
        with open(PurePath(".cache") / "signals" / "signals.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["timestamp"], "2024-01-12T00:00:00")
            self.assertEqual(list(data["signals"]["buy_long"].keys()), ["AAPL", "MSFT"])
            self.assertEqual(list(data["signals"]["sell_long"].keys()), ["GOOG", "AMZN"])

        # Now, try storing everything and check if they are stored
        renderer = JSONRenderer(store_portfolio=True, store_broker=True, store_account=True)
        renderer.render(state, PurePath(".cache"))

        with open(PurePath(".cache") / "signals" / "signals.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["timestamp"], "2024-01-12T00:00:00")
            self.assertEqual(list(data["signals"]["buy_long"].keys()), ["AAPL", "MSFT"])
            self.assertEqual(list(data["signals"]["sell_long"].keys()), ["GOOG", "AMZN"])
            self.assertEqual(data.keys(), {"timestamp", "signals", "portfolio", "broker", "account"})

            # Check that keys aren't empty
            self.assertTrue(data["portfolio"])
            self.assertTrue(data["broker"])
            self.assertTrue(data["account"])

from backtest.src.broker import Broker, MarginCall, BrokerState, StepState
from unittest import TestCase

class TestMarginCall(TestCase):
    def test_export(self):
        mc = MarginCall(5, 10_000)
        self.assertEqual(mc.export(), {"type": "MarginCall", "time_remaining": 5, "amount": 10_000})

    def test_load(self):
        mc = MarginCall.load({"type": "MarginCall", "time_remaining": 5, "amount": 10_000})
        self.assertEqual(mc.time_remaining, 5)
        self.assertEqual(mc.amount, 10_000)


    def test_eq(self):
        mc = MarginCall(5, 10_000)
        self.assertEqual(mc, 5)
        self.assertEqual(mc, MarginCall(5, 10_000))


class TestBrokerState(TestCase):
    def test_export(self):
        mc = MarginCall(5, 10_000)
        bs = BrokerState({"mc": mc}, False)
        state = bs.export()
        self.assertEqual(state, {
            "type": "BrokerState",
            "margin_calls": {
                "mc": {"type": "MarginCall", "time_remaining": 5, "amount": 10_000}
            },
            "bankruptcy": False})

    def test_load(self):
        mc = MarginCall(5, 10_000)
        bs = BrokerState({"mc": mc}, False)

        new_bs = BrokerState.load({
            "type": "BrokerState",
            "margin_calls": {
                "mc": {"type": "MarginCall", "time_remaining": 5, "amount": 10_000}
            },
            "bankruptcy": False})
        self.assertEqual(bs, new_bs)
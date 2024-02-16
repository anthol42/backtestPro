import numpy as np

from backtest.src.broker import Broker, MarginCall, BrokerState, StepState
from backtest.src.portfolio import Portfolio, Position
from backtest.src.account import Account
from backtest.src.tsData import DividendFrequency
from backtest.src.trade import TradeOrder, TradeType, BuyLongOrder, SellLongOrder, BuyShortOrder, SellShortOrder, TradeOrder
from unittest import TestCase
from datetime import datetime

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


class TestStepState(TestCase):
    def test_export(self):
        mc = MarginCall(5, 10_000)
        pending_orders = [
            TradeOrder(datetime(2021, 1, 1), "AAPL", (None, 100), 50,
                       50, TradeType.BuyLong, datetime(2021, 1, 1)),
            TradeOrder(datetime(2021, 1, 1), "TSLA", (None, 400), 25,
                       10, TradeType.BuyLong, datetime(2021, 1, 2))
        ]
        filled_orders = [
            TradeOrder(datetime(2021, 1, 1), "MSFT", (None, 250), 50,
                       50, TradeType.BuyLong, datetime(2021, 1, 1)),
            TradeOrder(datetime(2021, 1, 1), "V", (125, None), 25,
                          10, TradeType.SellLong, datetime(2021, 1, 2))
        ]
        ss = StepState(datetime(2021, 1, 1), 100_000, pending_orders, filled_orders,
                       {"mc": mc})

        state = ss.export()
        expected = {
            "type": "StepState",
            "timestamp": str(datetime(2021, 1, 1)),
            "worth": 100_000,
            "pending_orders": [order.export() for order in pending_orders],
            "filled_orders": [order.export() for order in filled_orders],
            "margin_calls": {key: value.export()
                             for key, value in {"mc": mc}.items()}
        }

        self.assertEqual(state, expected)

    def test_load(self):
        mc = MarginCall(5, 10_000)
        pending_orders = [
            TradeOrder(datetime(2021, 1, 1), "AAPL", (None, 100), 50,
                       50, TradeType.BuyLong, datetime(2021, 1, 1)),
            TradeOrder(datetime(2021, 1, 1), "TSLA", (None, 400), 25,
                       10, TradeType.BuyLong, datetime(2021, 1, 2))
        ]
        filled_orders = [
            TradeOrder(datetime(2021, 1, 1), "MSFT", (None, 250), 50,
                       50, TradeType.BuyLong, datetime(2021, 1, 1)),
            TradeOrder(datetime(2021, 1, 1), "V", (125, None), 25,
                       10, TradeType.SellLong, datetime(2021, 1, 2))
        ]
        expected = StepState(datetime(2021, 1, 1), 100_000, pending_orders, filled_orders,
                       {"mc": mc})
        state = {
            "type": "StepState",
            "timestamp": str(datetime(2021, 1, 1)),
            "worth": 100_000,
            "pending_orders": [order.export() for order in pending_orders],
            "filled_orders": [order.export() for order in filled_orders],
            "margin_calls": {key: value.export()
                             for key, value in {"mc": mc}.items()}
        }
        self.assertEqual(StepState.load(state), expected)


class TestBroker(TestCase):
    def test_init(self):
        account = Account(100_000)
        # Verify that the Broker doesn't raise any errors if the initial state is valid
        broker = Broker(account, 6.99, margin_interest=0.02)
        # Verify that the broker raise the right errors in the right conditions
        self.assertRaises(ValueError, Broker, account, 6.99, relative_commission=0.01)
        self.assertRaises(ValueError, Broker, account, 6.99, margin_interest=-0.01)
        self.assertRaises(ValueError, Broker, account, margin_interest=0.01, relative_commission=1.01)
        self.assertRaises(ValueError, Broker, account, margin_interest=0.01, relative_commission=-0.5)
        self.assertRaises(ValueError, Broker, account, commission=-5, margin_interest=0.01)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin=-0.01)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin=0.5, min_maintenance_margin=-0.1)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin=0.5, min_maintenance_margin=0.75)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin_short=-0.1)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin_short=0.5, min_maintenance_margin_short=-0.1)
        self.assertRaises(ValueError, Broker, account, commission=5, min_initial_margin_short=0.5, min_maintenance_margin_short=0.75)
        self.assertRaises(ValueError, Broker, account, commission=5, liquidation_delay=-1)

        # Test if _debt_record is correctly linked by reference to portfolio
        broker._debt_record["AAPL"] = 100
        self.assertEqual(broker.portfolio._debt_record["AAPL"], 100)

    def test_order_methods(self):
        """
        This test tests the method that creates orders such as:
            - buy_long
            - buy_short
            - sell_long
            - sell_short
        """
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))
        # Test buy_long
        broker.buy_long("AAPL", 100, 100, datetime(2021, 1, 15),
                        price_limit=(None, 100))
        expected = BuyLongOrder(datetime(2021, 1, 1),"AAPL", (None, 100.),
                                100, 100, datetime(2021, 1, 15))
        self.assertEqual(broker._queued_trade_offers[0], expected)

        # Test sell_long
        broker.sell_long("AAPL", 100, 100, datetime(2021, 1, 15),
                        price_limit=(95, 150))
        expected = SellLongOrder(datetime(2021, 1, 1),"AAPL", (95, 150),
                                100, 100, datetime(2021, 1, 15))
        self.assertEqual(broker._queued_trade_offers[1], expected)

        # Test buy_short
        broker.buy_short("AAPL", 0, 100, datetime(2021, 1, 15),
                        price_limit=(50, 100))
        expected = BuyShortOrder(datetime(2021, 1, 1),"AAPL", (50, 100),
                                0, 100, datetime(2021, 1, 15))
        self.assertEqual(broker._queued_trade_offers[2], expected)

        # Test sell_short
        broker.sell_short("AAPL", 0, 100, datetime(2021, 1, 15),
                        price_limit=(150, None))
        expected = SellShortOrder(datetime(2021, 1, 1),"AAPL", (150, None),
                                0, 100, datetime(2021, 1, 15))
        self.assertEqual(broker._queued_trade_offers[3], expected)

    def test_compute_dividend_payout(self):
        account = Account(100_000)
        broker = Broker(account, 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))
        trade = BuyLongOrder(datetime(2021, 1, 1),"AAPL", (None, 100.),
                                100, 100, datetime(2021, 1, 15)).\
                 convertToTrade(95., datetime(2021, 1, 2), "1")

        broker.portfolio.trade(trade)
        broker.portfolio.update_time_stock_idx(30)    # Held position for 30 days before dividends
        pos, _ = broker.portfolio["AAPL"]
        self.assertEqual(broker.compute_dividend_payout(pos, DividendFrequency.QUARTERLY, 0.24), 16)
        self.assertEqual(broker.compute_dividend_payout(pos, DividendFrequency.BIANNUALLY, 0.24), 8)
        self.assertAlmostEqual(broker.compute_dividend_payout(pos, DividendFrequency.YEARLY, 0.24), 3.945205479)
        self.assertEqual(broker.compute_dividend_payout(pos, DividendFrequency.MONTHLY, 0.24), 48)

    def test_new_margin_call(self):
        account = Account(100_000)
        broker = Broker(account, 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Test new margin call
        broker.new_margin_call(50_000, "Missing fund")
        self.assertEqual(broker.message.margin_calls["Missing fund"].amount, 50_000)

        broker.new_margin_call(30_000, "Missing fund")
        self.assertEqual(broker.message.margin_calls["Missing fund"].amount, 80_000)
        self.assertEqual(broker._debt_record["Missing fund"], 80_000)

    def test_remover_margin_call(self):
        account = Account(100_000)
        broker = Broker(account, 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Test remove margin call
        broker.new_margin_call(50_000, "Missing fund")
        broker.remove_margin_call("Missing fund")
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {})
    def test_get_short_collateral(self):
        account = Account(100_000)
        broker = Broker(account, 10, margin_interest=0.02, min_maintenance_margin_short=0.25)
        broker.set_current_timestamp(datetime(2021, 1, 10))

        prices = np.array([[]], dtype=np.float32)

        # Test without any short position
        self.assertEqual(broker._get_short_collateral(account.get_total_cash(), [], prices), 0)

        # Test with short positions NO margin call
        sec_names = ["AAPL", "TSLA", "MSFT"]
        prices = np.array([
            [102, 104, 98, 100],
            [202, 208, 196, 200],
            [302, 304, 298, 300]
                           ], dtype=np.float32)
        # Add positions to the portfolio
        broker.portfolio._short = {
            "AAPL": Position("AAPL", 100, False,125,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 12 512.5
            "TSLA": Position("TSLA", 100, False,203,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 25 012.5
            "MSFT":Position("MSFT", 100, False,300,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 37 512.5
        }
        self.assertEqual(broker._get_short_collateral(account.get_total_cash(), sec_names, prices), 75_037.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["AAPL"], 12_512.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["TSLA"], 25_012.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["MSFT"], 37_512.5)

        # We will change the prices to make a margin call
        prices = np.array([
            [102, 104, 98, 200],    # Collateral: 25 012.5
            [202, 208, 196, 300],   # Collateral: 37 512.5
            [302, 304, 298, 500]    # Collateral: 62 512.5
                           ], dtype=np.float32)
        self.assertEqual(broker._get_short_collateral(account.get_total_cash(), sec_names, prices), 100_000)
        self.assertEqual(broker.message.margin_calls["short margin call"].amount, 25_037.5)
        self.assertEqual(broker._debt_record["short margin call"], 25_037.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["AAPL"], 25_012.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["TSLA"], 37_512.5)
        self.assertEqual(broker._cache["short_collateral_contribution"]["MSFT"], 62_512.5)

        # Now make a short test wit relative commission
        broker = Broker(account, margin_interest=0.02, min_maintenance_margin_short=0.25, relative_commission=0.01)
        broker.set_current_timestamp(datetime(2021, 1, 10))
        # Add positions to the portfolio
        broker.portfolio._short = {
            "AAPL": Position("AAPL", 100, False,125,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 25 250
            "TSLA": Position("TSLA", 100, False,203,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 37 875
            "MSFT":Position("MSFT", 100, False,300,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 63 125
        }
        self.assertEqual(broker._get_short_collateral(account.get_total_cash(), sec_names, prices), 100_000)
        self.assertEqual(broker.message.margin_calls["short margin call"].amount, 26_250)
        self.assertEqual(broker._debt_record["short margin call"], 26_250)
        self.assertEqual(broker._cache["short_collateral_contribution"]["AAPL"], 25_250)
        self.assertEqual(broker._cache["short_collateral_contribution"]["TSLA"], 37_875)
        self.assertEqual(broker._cache["short_collateral_contribution"]["MSFT"], 63_125)

    def test_get_worth(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Add things to portfolio
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 100, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "MSFT": Position("MSFT", 100, True, 300,
                             datetime(2021, 1, 1), ratio_owned=0.75)
        }
        broker._debt_record["AAPL"] = 5000
        broker._debt_record["MSFT"] = 7500

        broker.portfolio._short = {
            "TSLA": Position("TSLA", 100, False, 200,
                             datetime(2021, 1, 1), ratio_owned=0.),
            "V": Position("V", 100, False, 150,
                             datetime(2021, 1, 1), ratio_owned=0.)
        }

        security_names = ["AAPL", "MSFT", "TSLA", "V"]
        prices = np.array([
            [102, 104, 98, 104],
            [253, 260, 247, 250],
            [302, 304, 298, 300],
            [78, 80, 70, 75]
                           ], dtype=np.float32)

        long_worth = 100*104 - 5000 + 100*250 - 7500    # 22 900
        short_worth = 100*300 + 100*75                  # 37 500
        expected_worth = 100_000 + long_worth - short_worth
        self.assertEqual(broker._get_worth(security_names, prices), expected_worth)

    def test_isMarginCall(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)

        is_mc, worth = broker._isMarginCall(100_000, 50_000, 0.25, 6.99, False)
        self.assertFalse(is_mc)
        self.assertEqual(worth, 0)
        is_mc, worth = broker._isMarginCall(100_000, 50_000, 0.25, 1.01, True)
        self.assertFalse(is_mc)
        self.assertEqual(worth, 0)

        is_mc, worth = broker._isMarginCall(60_000, 50_000, 0.25, 6.99, False)
        self.assertTrue(is_mc)
        self.assertAlmostEqual(worth, 5_005.2425)

        is_mc, worth = broker._isMarginCall(60_000, 50_000, 0.25, 1.01, True)
        self.assertTrue(is_mc)
        self.assertEqual(worth, 5_450)

    def test_get_long_collateral(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Test without any long position
        security_names = []
        prices = np.array([[]], dtype=np.float32)
        collateral = broker._get_long_collateral(100_000, security_names, prices)
        self.assertEqual(collateral, 0)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {})

        # Test with long positions NO margin call
        security_names = ["AAPL", "TSLA", "MSFT"]
        prices = np.array([
            [102, 104, 98, 100],
            [75, 76, 74, 83.5],
            [302, 304, 298, 300]
                           ], dtype=np.float32)
        # Add positions to the portfolio
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 100, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
            "TSLA": Position("TSLA", 100, True, 250,
                             datetime(2021, 1, 1), ratio_owned=0.75),    # Collateral: 0
            "MSFT": Position("MSFT", 100, True, 275,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
        }
        # Set debt record
        broker._debt_record["AAPL"] = 5000
        broker._debt_record["TSLA"] = 6250
        broker._debt_record["MSFT"] = 13750
        self.assertEqual(broker._get_long_collateral(100_000, security_names, prices), 0)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750
        })

        # We will change the prices to make a need for collateral
        prices = np.array([
            [102, 104, 98, 100],    # Collateral: 0
            [61, 60, 63, 62.5],    # Collateral: 1567.7425
            [302, 304, 298, 300]    # Collateral: 0
                           ], dtype=np.float32)
        self.assertAlmostEqual(broker._get_long_collateral(100_000, security_names, prices), 1567.7425)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750
        })
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["TSLA"], 1567.7425)

        # We will change the prices to make a margin call
        prices = np.array([
            [10.5, 12, 9, 10],    # Collateral: 4255.2425
            [61, 60, 63, 62.5],    # Collateral: 1567.7425
            [155, 150, 166, 160]    # Collateral: 1755.2425
                           ], dtype=np.float32)
        self.assertEqual(broker._get_long_collateral(5000, security_names, prices), 5000)
        self.assertAlmostEqual(broker.message.margin_calls["long margin call TSLA"].amount, 822.985)
        self.assertAlmostEqual(broker._debt_record["long margin call TSLA"], 822.985)
        self.assertAlmostEqual(broker.message.margin_calls["long margin call MSFT"].amount, 1755.2425)
        self.assertAlmostEqual(broker._debt_record["long margin call MSFT"], 1755.2425)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["AAPL"], 4255.2425)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["TSLA"], 744.7575)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["MSFT"], 0)

        # Now test with relative commission
        broker = Broker(Account(100_000), relative_commission=0.02, margin_interest=0.02)
        security_names = ["AAPL", "TSLA", "MSFT"]
        prices = np.array([
            [10.5, 12, 9, 10],    # Collateral: 4265
            [61, 60, 63, 62.5],    # Collateral: 1656.25
            [155, 150, 166, 160]    # Collateral: 1990
                           ], dtype=np.float32)
        # Add positions to the portfolio
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 100, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
            "TSLA": Position("TSLA", 100, True, 250,
                             datetime(2021, 1, 1), ratio_owned=0.75),    # Collateral: 0
            "MSFT": Position("MSFT", 100, True, 275,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
        }
        # Set debt record
        broker._debt_record["AAPL"] = 5000
        broker._debt_record["TSLA"] = 6250
        broker._debt_record["MSFT"] = 13750
        self.assertEqual(broker._get_long_collateral(5000, security_names, prices), 5000)
        self.assertAlmostEqual(broker.message.margin_calls["long margin call TSLA"].amount, 921.25)
        self.assertAlmostEqual(broker._debt_record["long margin call TSLA"], 921.25)
        self.assertAlmostEqual(broker.message.margin_calls["long margin call MSFT"].amount, 1990)
        self.assertAlmostEqual(broker._debt_record["long margin call MSFT"], 1990)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["AAPL"], 4265)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["TSLA"], 735)
        self.assertAlmostEqual(broker._cache["long_collateral_contribution"]["MSFT"], 0)

    def test_update_acount_collateral(self):
        broker = Broker(Account(75_000), 10., margin_interest=0.02, min_maintenance_margin_short=0.25)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Test without any long position
        security_names = []
        prices = np.array([[]], dtype=np.float32)
        broker._update_account_collateral(datetime(2021, 1, 1), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 75_000)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {})

        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],    # AAPL
            [61, 60, 63, 62.5],     # TSLA
            [302, 304, 298, 300],   # MSFT
            [102, 104, 98, 100],    # V
            [202, 208, 196, 200],   # CAT
            [302, 304, 298, 300]    # OLN
                           ], dtype=np.float32)

        # Add long positions
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 100, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
            "TSLA": Position("TSLA", 100, True, 250,
                             datetime(2021, 1, 1), ratio_owned=0.75),    # Collateral: 1570
            "MSFT": Position("MSFT", 100, True, 275,
                             datetime(2021, 1, 1), ratio_owned=0.5),    # Collateral: 0
        }
        # Set debt record
        broker._debt_record["AAPL"] = 5000
        broker._debt_record["TSLA"] = 6250
        broker._debt_record["MSFT"] = 13750

        # Add short positions
        broker.portfolio._short = {
            "V": Position("V", 100, False,125,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 12 512.5
            "CAT": Position("CAT", 100, False,203,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 25 012.5
            "OLN":Position("OLN", 100, False,300,
                             datetime(2021, 1, 1), ratio_owned=0),    # Collateral: 37 512.5
        }

        # Test
        broker._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertEqual(broker.message.margin_calls, {"short margin call": MarginCall(2, 37.5),
                         "long margin call TSLA": MarginCall(2, 1570)})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
            "short margin call": 37.5,
            "long margin call TSLA": 1570
        })


        # Test with no short margin call
        prices = np.array([
            [102, 104, 98, 100],    # AAPL
            [61, 60, 63, 62.5],     # TSLA
            [302, 304, 298, 300],   # MSFT
            [102, 104, 98, 100],    # V
            [202, 208, 196, 200],   # CAT
            [302, 304, 298, 298]    # OLN: Collateral - 37262.5
                           ], dtype=np.float32)
        # After short, I have 212.5$ available.

        broker._update_account_collateral(datetime(2021, 1, 3), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertEqual(broker.message.margin_calls, {"long margin call TSLA": MarginCall(2, 1357.5)})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
            "long margin call TSLA": 1357.5
        })

        # Test with no short margin call
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [83, 84, 82, 83],  # TSLA    # Collateral: 32.5
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 298]  # OLN: Collateral - 37262.5
        ], dtype=np.float32)
        # After short, I have 212.5$ available.  Then, we substract the collateral from long position and we get:
        # 212.5 - 32.5 = 180$
        broker._update_account_collateral(datetime(2021, 1, 4), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 180)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
        })

        # Now make a simple test with relative commission
        broker = Broker(Account(75_000), relative_commission=0.01, margin_interest=0.02, min_maintenance_margin_short=0.25)
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)

        # Add long positions
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 100, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),  # Collateral: 0
            "TSLA": Position("TSLA", 100, True, 250,
                             datetime(2021, 1, 1), ratio_owned=0.75),  # Collateral: 1609.375
            "MSFT": Position("MSFT", 100, True, 275,
                             datetime(2021, 1, 1), ratio_owned=0.5),  # Collateral: 0
        }
        # Set debt record
        broker._debt_record["AAPL"] = 5000
        broker._debt_record["TSLA"] = 6250
        broker._debt_record["MSFT"] = 13750

        # Add short positions
        broker.portfolio._short = {
            "V": Position("V", 100, False, 125,
                          datetime(2021, 1, 1), ratio_owned=0),  # Collateral: 12 625
            "CAT": Position("CAT", 100, False, 203,
                            datetime(2021, 1, 1), ratio_owned=0),  # Collateral: 25 250
            "OLN": Position("OLN", 100, False, 300,
                            datetime(2021, 1, 1), ratio_owned=0),  # Collateral: 37 875
        }
        broker._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertEqual(broker.message.margin_calls, {"short margin call": MarginCall(2, 750),
                                                      "long margin call TSLA": MarginCall(2, 1609.375)})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
            "short margin call": 750,
            "long margin call TSLA": 1609.375
        })

    def test_get_buy_price(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        # Test with no limit
        price = (100, 102, 98, 101)
        self.assertEqual(broker._get_buy_price(None, None, price), 100)

        # Test with lower limit
        # Order pass at limit
        self.assertEqual(broker._get_buy_price(99, None, price), 99)
        # Order doesn't pass at limit
        self.assertEqual(broker._get_buy_price(97, None, price), None)
        # Order pass at open
        self.assertEqual(broker._get_buy_price(101, None, price), 100)

        # Test with upper limit
        # Order pass at limit
        self.assertEqual(broker._get_buy_price(None, 101, price), 101)
        # Order doesn't pass at limit
        self.assertEqual(broker._get_buy_price(None, 103, price), None)
        # Order pass at open
        self.assertEqual(broker._get_buy_price(None, 99, price), 100)

        # Test with both limits
        # Open price is higher than high --> Should return open price
        self.assertEqual(broker._get_buy_price(98.5, 99, price), 100)
        # Open price is lower than low --> Should return open price
        self.assertEqual(broker._get_buy_price(101, 105, price), 100)
        # Price range gets higher than high and lower than low --> Should return high
        self.assertEqual(broker._get_buy_price(98.5, 101, price), 101)
        # Price range gets lower than low --> Should return Low
        self.assertEqual(broker._get_buy_price(98.5, 105, price), 98.5)
        # price range gets higher than high --> Should return high
        self.assertEqual(broker._get_buy_price(97, 101, price), 101)
        # price range is not within the limits --> Should return None
        self.assertEqual(broker._get_buy_price(97, 105, price), None)

    def test_get_sell_price(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        # Test with no limit
        price = (100, 102, 98, 101)
        self.assertEqual(broker._get_sell_price(None, None, price), 100)

        # Test with lower limit
        # Order pass at limit
        self.assertEqual(broker._get_sell_price(99, None, price), 99)
        # Order doesn't pass at limit
        self.assertEqual(broker._get_sell_price(97, None, price), None)
        # Order pass at open
        self.assertEqual(broker._get_sell_price(101, None, price), 100)

        # Test with upper limit
        # Order pass at limit
        self.assertEqual(broker._get_sell_price(None, 101, price), 101)
        # Order doesn't pass at limit
        self.assertEqual(broker._get_sell_price(None, 103, price), None)
        # Order pass at open
        self.assertEqual(broker._get_sell_price(None, 99, price), 100)

        # Test with both limits
        # Open price is higher than high --> Should return open price
        self.assertEqual(broker._get_sell_price(98.5, 99, price), 100)
        # Open price is lower than low --> Should return open price
        self.assertEqual(broker._get_sell_price(101, 105, price), 100)
        # Price range gets higher than high and lower than low --> Should return low
        self.assertEqual(broker._get_sell_price(98.5, 101, price), 98.5)
        # Price range gets lower than low --> Should return low
        self.assertEqual(broker._get_sell_price(98.5, 105, price), 98.5)
        # price range gets higher than high --> Should return high
        self.assertEqual(broker._get_sell_price(97, 101, price), 101)
        # price range is not within the limits --> Should return None
        self.assertEqual(broker._get_sell_price(97, 105, price), None)

    def test_make_trade_BL(self):
        """
        Test the make trade method for a buy long order with relative and absolute commissions.
        """
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],    # AAPL
            [61, 60, 63, 62.5],     # TSLA
            [302, 304, 298, 300],   # MSFT
            [102, 104, 98, 100],    # V
            [202, 208, 196, 200],   # CAT
            [302, 304, 298, 300]    # OLN
                           ], dtype=np.float32)

        # Try to buy a non-marginable security on margin.
        # (Should do nothing since the security might becom marginable later)
        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.01)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 100,
                           100, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                          marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.portfolio._long, {})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._long, {})
        self.assertEqual(broker_rel.account.get_cash(), 100_000)
        self.assertEqual(broker_rel._debt_record, {})

        # Try to buy it with no limit and no margin
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 100,
                           0, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                   102,
                                                                   datetime(2021, 1, 2), 1.)})
        self.assertEqual(broker_rel.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                   102,
                                                                   datetime(2021, 1, 2), 1.)})
        self.assertEqual(broker_abs.account.get_cash(), 89793.01)
        self.assertEqual(broker_rel.account.get_cash(), 89698)
        self.assertEqual(broker_abs._debt_record, {"AAPL": 0})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 0})

        # Test with minimum margin
        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.01)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 100,
                           100, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 200, True,
                                                                   102,
                                                                   datetime(2021, 1, 2), 0.5)})
        self.assertEqual(broker_rel.portfolio._long, {"AAPL": Position("AAPL", 200, True,
                                                                   102,
                                                                   datetime(2021, 1, 2), 0.5)})
        self.assertEqual(broker_abs.account.get_cash(), 89793.01)
        self.assertEqual(broker_rel.account.get_cash(), 89698)
        self.assertEqual(broker_abs._debt_record, {"AAPL": 10200})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 10302})

        # Test with smaller margin than authorized
        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.01)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 100,
                           101, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertRaises(RuntimeError, broker_abs.make_trade, order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True)
        self.assertRaises(RuntimeError, broker_rel.make_trade, order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True)

        # Try to buy if not enough cash
        broker_abs = Broker(Account(10_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(10_000), margin_interest=0.02, relative_commission=0.01)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 98,
                           98, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._long, {})
        self.assertEqual(broker_rel.portfolio._long, {})
        self.assertEqual(broker_abs.account.get_cash(), 10_000)
        self.assertEqual(broker_rel.account.get_cash(), 10_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel._debt_record, {})

        # Now, we will test with limit orders
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.set_current_timestamp(datetime(2021, 1, 1))
        # Test with superior limit.  We want to buy at that price or lower. So we buy only if the price has been lower
        # during the day
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (100, None), 100,
                           100, TradeType.BuyLong, datetime(2021, 1, 1))

        price = np.array([101, 102, 100.0001,101])    # OHLC
        self.assertFalse(broker.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker.portfolio._long, {})
        self.assertEqual(broker.account.get_cash(), 100_000)
        self.assertEqual(broker._debt_record, {})

        price = np.array([101, 102, 99.999999, 101])
        self.assertTrue(broker.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))

        self.assertEqual(broker.portfolio._long, {"AAPL": Position("AAPL", 200, True,
                                                                   100,
                                                                   datetime(2021, 1, 2), 0.5)})
        self.assertEqual(broker.account.get_cash(), 89993.01)
        self.assertEqual(broker._debt_record, {"AAPL": 10_000})

        # If open is under the limit, we buy at open
        price = np.array([99, 102, 100.0001, 101])
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        self.assertTrue(broker.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker.portfolio._long, {"AAPL": Position("AAPL", 200, True,
                                                                   99,
                                                                   datetime(2021, 1, 2), 0.5)})
        self.assertEqual(broker.account.get_cash(), 90093.01)
        self.assertEqual(broker._debt_record, {"AAPL": 9900})

    def test_make_trade_SL(self):
        """
        Test the make trade method for a sell long order with relative and absolute commissions.
        """
        # Try to sell a position that doesn't exist
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],    # AAPL
            [61, 60, 63, 62.5],     # TSLA
            [302, 304, 298, 300],   # MSFT
            [102, 104, 98, 100],    # V
            [202, 208, 196, 200],   # CAT
            [302, 304, 298, 300]    # OLN
                           ], dtype=np.float32)

        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.01)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 100,
                           100, TradeType.BuyLong, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.portfolio._long, {})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})

        # Now add a position to the portfolio
        broker_abs.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5)
        }
        broker_rel.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5)
        }
        broker_abs._debt_record["AAPL"] = 10_000
        broker_rel._debt_record["AAPL"] = 10_000

        # Try to sell more shares than we have
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 201,
                           0, TradeType.SellLong, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))

        # Try to sell with gains
        broker_abs._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        broker_rel._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        price = (106, 115, 105, 107)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, 110), 100,
                           0, TradeType.SellLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.account.get_cash(), 105_993.01)
        self.assertEqual(broker_rel.account.get_cash(), 105_890)
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_rel.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs._debt_record, {"AAPL": 5_000})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 5_000})

        # Try to sell with no gains and the debt will make a little loss
        price = (50, 51, 49, 51)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (50, 110), 100,
                           0, TradeType.SellLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertAlmostEqual(broker_abs.account.get_cash(), 105_986.02)
        self.assertEqual(broker_rel.account.get_cash(), 105_840)
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 0, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_rel.portfolio._long, {"AAPL": Position("AAPL", 0, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs._debt_record, {"AAPL": 0})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 0})

        # Sell at loss and update collateral before selling
        # the stock if there was one.
        price = (50, 51, 49, 51)
        broker_abs.account._cash = 2500
        broker_rel.account._cash = 2500
        broker_abs.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5)
        }
        broker_rel.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5)
        }
        broker_abs._debt_record["AAPL"] = 10_000
        broker_rel._debt_record["AAPL"] = 10_000

        # Now, we should update the collateral and there should have 2355.2425$ of collateral for the long position.
        # However, we only have 2500$ in our account.  So, we should have 144.7575 left for AAPL for abs
        # For rel: we should have 2426.5$ in collateral and 73.5$ left in the account.
        prices[0] = price
        broker_abs._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        broker_rel._update_account_collateral(datetime(2021, 1, 2), security_names, prices)

        # Now, by selling our position at stoploss price: 50$, we will create a margin call for missing funds.
        # For abs: we should have 6.99$ in margin call
        # For rel: we should have 75$ in margin call
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (50, 110), 100,
                           0, TradeType.SellLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.account.get_cash(), 2493.01)
        self.assertEqual(broker_rel.account.get_cash(), 2450)
        self.assertEqual(broker_abs.message.margin_calls, {})
        self.assertEqual(broker_rel.message.margin_calls, {})
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 100, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs._debt_record, {"AAPL": 5000})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 5000})


        # Now test the same thing with a margin call
        broker_abs.account._cash = 0
        broker_rel.account._cash = 0

        # Now, we will update the collateral. since our account is empty, all the collateral will be put as a margin
        # call.
        # For abs: we should have 1255.2425$ in margin call
        # For rel: we should have 1287.5$ in margin call
        prices[0] = (51, 52, 49, 50)
        broker_abs._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        broker_rel._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertAlmostEqual(broker_abs.message.margin_calls["long margin call AAPL"].amount, 1255.2425)
        self.assertAlmostEqual(broker_rel.message.margin_calls["long margin call AAPL"].amount, 1287.5)

        # We will sell the position and check if the margin call is removed.
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (50, 110), 100,
                           0, TradeType.SellLong, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(price), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.account.get_cash(), 0)
        self.assertEqual(broker_abs.account.get_cash(), 0)
        self.assertEqual(list(broker_abs.message.margin_calls.keys()), ["missing_funds"])
        self.assertEqual(list(broker_rel.message.margin_calls.keys()), ["missing_funds"])
        self.assertAlmostEqual(broker_abs.message.margin_calls["missing_funds"].amount, 6.99)
        self.assertAlmostEqual(broker_rel.message.margin_calls["missing_funds"].amount, 50)
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 0, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs.portfolio._long, {"AAPL": Position("AAPL", 0, True,
                                                                       100,
                                                                       datetime(2021, 1, 1),
                                                                       0.5)})
        self.assertEqual(broker_abs._debt_record, {"AAPL": 0, "missing_funds": 6.989999999999782})
        self.assertEqual(broker_rel._debt_record, {"AAPL": 0, "missing_funds": 50})

    def test_make_trade_SS(self):
        """
        Test the make trade method for a sell short order with relative and absolute commissions.
        """
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)

        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.02)
        # Start by trying to sell a position that is not shortable
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 0,
                           100, TradeType.SellShort, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.portfolio._short, {})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._short, {})
        self.assertEqual(broker_rel.account.get_cash(), 100_000)
        self.assertEqual(broker_rel._debt_record, {})

        # Now we will try to sell short a position with less margin than authorized
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (100, None), 0,
                           800, TradeType.SellShort, datetime(2021, 1, 1))
        self.assertRaises(RuntimeError, broker_abs.make_trade, order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True)
        self.assertRaises(RuntimeError, broker_rel.make_trade,order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True)
        self.assertEqual(broker_abs.portfolio._short, {})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._short, {})
        self.assertEqual(broker_rel.account.get_cash(), 100_000)
        self.assertEqual(broker_rel._debt_record, {})

        # Now, we will sell short a position that we can sell
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (100, None), 0,
                           100, TradeType.SellShort, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._short, {"AAPL": Position("AAPL", 100, False,
                                                                       100,
                                                                       datetime(2021, 1, 2),
                                                                       0.)})
        self.assertEqual(broker_rel.portfolio._short, {"AAPL": Position("AAPL", 100, False,
                                                                       100,
                                                                       datetime(2021, 1, 2),
                                                                       0.)})
        self.assertAlmostEqual(broker_abs.account.get_cash(), 97_484.2725)
        self.assertAlmostEqual(broker_rel.account.get_cash(), 97_050)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel._debt_record, {})
        self.assertEqual(broker_abs.message.margin_calls, {})
        self.assertEqual(broker_rel.message.margin_calls, {})

        # Test with limit order that doesn't pass
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, 105), 0,
                           100, TradeType.SellShort, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))

    def test_make_trade_BS(self):
        """
        Test the make trade method for a buy short orders with relative and absolute commissions.
        """
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)
        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.02)

        # Try to buy a something that we do not have any position opened
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 0,
                           100, TradeType.BuyShort, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.portfolio._short, {})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._short, {})
        self.assertEqual(broker_rel.account.get_cash(), 100_000)
        self.assertEqual(broker_rel._debt_record, {})

        # Try to buy a position without enough shares
        broker_abs.portfolio._short = {
            "AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)
        }
        broker_rel.portfolio._short = {
            "AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)
        }
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (None, None), 0,
                           200, TradeType.BuyShort, datetime(2021, 1, 1))
        self.assertFalse(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertFalse(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=False, shortable=False))
        self.assertEqual(broker_abs.portfolio._short, {"AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_abs.account.get_cash(), 100_000)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._short, {"AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_rel.account.get_cash(), 100_000)
        self.assertEqual(broker_rel._debt_record, {})

        # Try to buy a position that should work
        broker_abs._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        broker_rel._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (100, None), 0,
                           100, TradeType.BuyShort, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._short, {"AAPL": Position("AAPL", 0, False,
                             100, datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_abs.account.get_cash(), 89_993.01)
        self.assertEqual(broker_abs._debt_record, {})
        self.assertEqual(broker_rel.portfolio._short, {"AAPL": Position("AAPL", 0, False,
                             100, datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_rel.account.get_cash(), 89_800)
        self.assertEqual(broker_rel._debt_record, {})

        # Try to sell short without enough money
        broker_abs.account._cash = 0
        broker_rel.account._cash = 0
        broker_abs.portfolio._short = {
            "AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)
        }
        broker_rel.portfolio._short = {
            "AAPL": Position("AAPL", 100, False, 100,
                             datetime(2021, 1, 1), ratio_owned=0)
        }
        broker_abs.portfolio._short_len = 1
        broker_rel.portfolio._short_len = 1
        broker_abs._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        broker_rel._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        # Now, we should have huge short margin calls
        # For abs: we should have 12 508.7375$ in margin call
        # For rel: we should have 12 750$ in margin call
        self.assertEqual(broker_abs.message.margin_calls["short margin call"].amount, 12508.7375)
        self.assertEqual(broker_rel.message.margin_calls["short margin call"].amount, 12750)
        order = TradeOrder(datetime(2021, 1, 1), security_names[0], (100, None), 0,
                           100, TradeType.BuyShort, datetime(2021, 1, 1))
        self.assertTrue(broker_abs.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertTrue(broker_rel.make_trade(order, security_price=tuple(prices[0]), timestamp=datetime(2021, 1, 2),
                            marginable=True, shortable=True))
        self.assertEqual(broker_abs.portfolio._short, {"AAPL": Position("AAPL", 0, False,
                             100, datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_abs.account.get_cash(), 0)
        self.assertEqual(broker_abs._debt_record, {"missing_funds": 10_006.99})
        self.assertEqual(broker_rel.portfolio._short, {"AAPL": Position("AAPL", 0, False,
                             100, datetime(2021, 1, 1), ratio_owned=0)})
        self.assertEqual(broker_rel.account.get_cash(), 0)
        self.assertEqual(broker_rel._debt_record, {"missing_funds": 10_200})


    def test_get_deltas(self):
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)
        # Start with the short version
        # Test with absolute commissions first
        broker_abs = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker_rel = Broker(Account(100_000), margin_interest=0.02, relative_commission=0.02)
        broker_abs.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 150, True, 150,
                                datetime(2021, 1, 1), ratio_owned=0.5),   # 4392.7425$ in collateral
            "MSFT": Position("MSFT", 100, True, 200,
                                datetime(2021, 1, 1), ratio_owned=1.)
        }
        broker_rel.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 150, True, 150,
                                datetime(2021, 1, 1), ratio_owned=0.5),   # 4524.75$ in collateral
            "MSFT": Position("MSFT", 100, True, 200,
                                datetime(2021, 1, 1), ratio_owned=1.)
        }
        broker_abs._debt_record = {
            "AAPL": 10_000,
            "TSLA": 11_250,
            "MSFT": 0
        }
        broker_rel._debt_record = {
            "AAPL": 10_000,
            "TSLA": 11_250,
            "MSFT": 0
        }
        broker_abs.portfolio._short = {
            "V": Position("V", 100, False, 100,
                          datetime(2021, 1, 1), ratio_owned=0.),
            "CAT": Position("CAT", 150, False, 150,
                          datetime(2021, 1, 1), ratio_owned=0.),
            "OLN": Position("OLN", 200, False, 200,
                          datetime(2021, 1, 1), ratio_owned=0.)
        }
        broker_rel.portfolio._short = {
            "V": Position("V", 100, False, 100,
                          datetime(2021, 1, 1), ratio_owned=0.),
            "CAT": Position("CAT", 150, False, 150,
                          datetime(2021, 1, 1), ratio_owned=0.),
            "OLN": Position("OLN", 200, False, 200,
                          datetime(2021, 1, 1), ratio_owned=0.)
        }
        expected = np.array([-7448.2525, -2423.2525, 5101.7475])
        np.testing.assert_array_almost_equal(broker_abs._get_deltas(10_000, security_names, prices), expected)

        # Test with relative commissions
        expected = np.array([-7399, -2273.5, 5402])
        np.testing.assert_array_almost_equal(broker_rel._get_deltas(10_000, security_names, prices), expected)

        # Then do the long version
        # Test with absolute commissions first

        expected = np.array([393.01, -7714.2475, 20_193.01])
        np.testing.assert_array_almost_equal(broker_abs._get_deltas(10_000, security_names, prices, short=False),
                                             expected)
        # Test with relative commissions
        expected = np.array([-8, -7758.25, 19596])
        np.testing.assert_array_almost_equal(broker_rel._get_deltas(10_000, security_names, prices, short=False),
                                             expected)

    def test_execute_trades(self):
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)
        next_prices = np.array([
            [103, 105, 99, 101],  # AAPL
            [60, 59, 62, 61.5],  # TSLA
            [300, 300, 295, 298],  # MSFT
            [103, 105, 99, 101],  # V
            [200, 202, 194, 199],  # CAT
            [295, 300, 290, 296]  # OLN
        ], dtype=np.float32)
        marginables = np.ones((6, 2), dtype=bool)
        broker.buy_long("AAPL", 100, 100, expiry=datetime(2021, 1, 5), price_limit=(None, None))
        broker.buy_long("TSLA", 100, 100, expiry=datetime(2021, 1, 2), price_limit=(50, None))
        broker.buy_long("MSFT", 100, 100, expiry=None, price_limit=(250, 350))
        filled_orders = broker._execute_trades(datetime(2021, 1, 2), security_names, next_prices, marginables)
        self.assertEqual(list(broker.portfolio._long.keys()), ["AAPL"])
        self.assertEqual([order.security for order in broker._queued_trade_offers], ["TSLA", "MSFT"])
        self.assertEqual(filled_orders, [TradeOrder(None, "AAPL",
                                                    (None, None), 100, 100,
                                                    TradeType.BuyLong,datetime(2021, 1, 5))])
        filled_orders = broker._execute_trades(datetime(2021, 1, 3), security_names, next_prices, marginables)
        self.assertEqual(list(broker.portfolio._long.keys()), ["AAPL"])
        self.assertEqual([order.security for order in broker._queued_trade_offers], ["MSFT"])
        self.assertEqual(filled_orders, [])
    def test_liquidate(self):
        """
        Test the liquidate method.
        """
        # Start with few case where liquidating short positions will be enough to cover
        security_names = ["AAPL", "TSLA", "MSFT", "V", "CAT", "OLN"]
        prices = np.array([
            [102, 104, 98, 100],  # AAPL
            [61, 60, 63, 62.5],  # TSLA
            [302, 304, 298, 300],  # MSFT
            [102, 104, 98, 100],  # V
            [202, 208, 196, 200],  # CAT
            [302, 304, 298, 300]  # OLN
        ], dtype=np.float32)
        next_prices = np.array([
            [103, 105, 99, 101],  # AAPL
            [60, 59, 62, 61.5],  # TSLA
            [300, 300, 295, 298],  # MSFT
            [103, 105, 99, 101],  # V
            [200, 202, 194, 199],  # CAT
            [295, 300, 290, 296]  # OLN
        ], dtype=np.float32)
        marginables = np.ones((6, 2), dtype=bool)
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 150, True, 150,
                                datetime(2021, 1, 1), ratio_owned=0.5),   # 4223.9925$ in collateral
            "MSFT": Position("MSFT", 100, True, 200,
                                datetime(2021, 1, 1), ratio_owned=1.)
        }
        broker.portfolio._long_len = 3
        broker._debt_record = {
            "AAPL": 10_000,
            "TSLA": 11_250,
            "MSFT": 0
        }
        broker.portfolio._short = {
            "V": Position("V", 100, False, 100,
                          datetime(2021, 1, 1), ratio_owned=0.),    # 12 758.7375 in collateral
            "CAT": Position("CAT", 150, False, 150,
                          datetime(2021, 1, 1), ratio_owned=0.),    # 37 883.7375 in collateral
            "OLN": Position("OLN", 200, False, 200,
                          datetime(2021, 1, 1), ratio_owned=0.)     # 75 508.7375
        }
        broker.portfolio._short_len = 3
        broker._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertAlmostEqual(broker._debt_record["short margin call"], 25026.2125)
        self.assertAlmostEqual(broker.message.margin_calls["short margin call"].amount, 25026.2125)

        # We should only sell the CAT position to cover the margin call
        broker._liquidate(3_000, datetime(2021, 1, 2), security_names, prices)
        broker._execute_trades(datetime(2021, 1, 2), security_names, next_prices, marginables)
        # Now, the whole short margin call should be erased.  And we should have 12 482.525$ remaining that should be
        # removed from collateral.  However, buying back the position will cost 30 006.99$.  So, we should have a
        # missing funds margin call of 17 524.465$.
        self.assertEqual(broker.portfolio._short, {
            "V": Position("V", 100, False, 100,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "CAT": Position("CAT", 0, False, 150,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "OLN": Position("OLN", 200, False, 200,
                            datetime(2021, 1, 1), ratio_owned=0.)
        })
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertAlmostEqual(broker._debt_record["missing_funds"], 17_524.465)
        self.assertAlmostEqual(broker.message.margin_calls["missing_funds"].amount, 17_524.465)

        # Now, try to liquidate for the real margin call: 25026.2125$.  The order of selling should be: OLN, V, CAT
        # Reset broker -- Add the trading fee so that buying short position can cover short margin call
        broker = Broker(Account(100_020.97), 6.99, margin_interest=0.02)
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 150, True, 150,
                                datetime(2021, 1, 1), ratio_owned=0.5),   #  4223.9925$ in collateral
            "MSFT": Position("MSFT", 100, True, 200,
                                datetime(2021, 1, 1), ratio_owned=1.)
        }
        broker.portfolio._long_len = 3
        broker._debt_record["AAPL"] = 10_000
        broker._debt_record["TSLA"] = 11_250
        broker._debt_record["MSFT"] = 0

        broker.portfolio._short = {
            "V": Position("V", 100, False, 100,
                          datetime(2021, 1, 1), ratio_owned=0.),    # 12 758.7375 in collateral
            "CAT": Position("CAT", 150, False, 150,
                          datetime(2021, 1, 1), ratio_owned=0.),    # 37 883.7375 in collateral
            "OLN": Position("OLN", 200, False, 200,
                          datetime(2021, 1, 1), ratio_owned=0.)     # 75 508.7375
        }
        broker.portfolio._short_len = 3
        broker._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertAlmostEqual(broker._debt_record["short margin call"], 25005.2425)
        self.assertAlmostEqual(broker._debt_record["short margin call"], 25005.2425)
        self.assertAlmostEqual(broker.message.margin_calls["long margin call TSLA"].amount, 4223.9925)
        self.assertAlmostEqual(broker._debt_record["long margin call TSLA"], 4223.9925)
        broker._liquidate(25005.2425, datetime(2021, 1, 2), security_names, prices)
        broker._execute_trades(datetime(2021, 1, 2), security_names, next_prices, marginables)
        # Collateral contribution - cost of the position:
        # V: 2501.7475; ; 12508.7375
        # CAT: 7501.7475; ; 37508.7375
        # OLN: 15001.7475; ; 75008.7375
        # Now, the whole short margin call should be erased and the short collateral should be erased too.
        # The total cash in the account should be 679.03$.  However, we still have the long margin call for TSLA that
        # should be 3544.9625$.
        self.assertEqual(broker.portfolio._short, {
            "V": Position("V", 0, False, 100,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "CAT": Position("CAT", 0, False, 150,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "OLN": Position("OLN", 0, False, 200,
                            datetime(2021, 1, 1), ratio_owned=0.)
        })
        self.assertAlmostEqual(broker.account.get_total_cash(), 4336.4925)
        self.assertAlmostEqual((broker.account.get_cash()), 0)
        self.assertAlmostEqual(broker._debt_record["missing_funds"], 3636.4925)
        self.assertAlmostEqual(broker.message.margin_calls["missing_funds"].amount, 3636.4925)


        # Try to liquidate also long positions

        # Reset broker
        broker = Broker(Account(100_000), 6.99, margin_interest=0.02)
        broker.portfolio._long = {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 150, True, 150,
                             datetime(2021, 1, 1), ratio_owned=0.5),  # 4223.9925$ in collateral
            "MSFT": Position("MSFT", 100, True, 200,
                             datetime(2021, 1, 1), ratio_owned=1.)
        }
        broker.portfolio._long_len = 3
        broker._debt_record["AAPL"] = 10_000
        broker._debt_record["TSLA"] = 11_250
        broker._debt_record["MSFT"] = 0

        broker.portfolio._short = {
            "V": Position("V", 100, False, 100,
                          datetime(2021, 1, 1), ratio_owned=0.),  # 12 758.7375 in collateral
            "CAT": Position("CAT", 150, False, 150,
                            datetime(2021, 1, 1), ratio_owned=0.),  # 37 883.7375 in collateral
            "OLN": Position("OLN", 200, False, 200,
                            datetime(2021, 1, 1), ratio_owned=0.)  # 75 508.7375
        }
        broker.portfolio._short_len = 3
        broker._update_account_collateral(datetime(2021, 1, 2), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertAlmostEqual(broker._debt_record["short margin call"], 25026.2125)
        self.assertAlmostEqual(broker.message.margin_calls["short margin call"].amount, 25026.2125)

        broker._liquidate(25026.2125, datetime(2021, 1, 2), security_names, prices)
        broker._execute_trades(datetime(2021, 1, 2), security_names, next_prices, marginables)
        self.assertEqual(broker.portfolio._short, {
            "V": Position("V", 0, False, 100,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "CAT": Position("CAT", 0, False, 150,
                            datetime(2021, 1, 1), ratio_owned=0.),
            "OLN": Position("OLN", 0, False, 200,
                            datetime(2021, 1, 1), ratio_owned=0.)
        })
        self.assertEqual(broker.portfolio._long, {
            "AAPL": Position("AAPL", 200, True, 100,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "TSLA": Position("TSLA", 0, True, 150,
                             datetime(2021, 1, 1), ratio_owned=0.5),
            "MSFT": Position("MSFT", 100, True, 200,
                             datetime(2021, 1, 1), ratio_owned=1.)
        })
        self.assertAlmostEqual(broker.account.get_total_cash(), 0)
        self.assertAlmostEqual((broker.account.get_cash()), 0)
        self.assertAlmostEqual(broker._debt_record["missing_funds"], 1577.96)
        self.assertAlmostEqual(broker.message.margin_calls["missing_funds"].amount, 1577.96)


        # Finally, test if we can get a bankruptcy
        # Reset broker and change next_prices

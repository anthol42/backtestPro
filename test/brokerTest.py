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
        self.assertEqual(broker.message.margin_calls["Missing fund_1"].amount, 30_000)
        self.assertEqual(broker._debt_record["Missing fund_1"], 30_000)
        self.assertEqual(broker._debt_record["Missing fund"], 50_000)

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


        # We will change the prices to make a margin call
        prices = np.array([
            [102, 104, 98, 200],    # Collateral: 25 012.5
            [202, 208, 196, 300],   # Collateral: 37 512.5
            [302, 304, 298, 500]    # Collateral: 62 512.5
                           ], dtype=np.float32)
        self.assertEqual(broker._get_short_collateral(account.get_total_cash(), sec_names, prices), 100_000)
        self.assertEqual(broker.message.margin_calls["short margin call"].amount, 25_037.5)
        self.assertEqual(broker._debt_record["short margin call"], 25_037.5)

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

        is_mc, worth = broker._isMarginCall(100_000, 50_000, 0.25)
        self.assertFalse(is_mc)
        self.assertEqual(worth, 0)

        is_mc, worth = broker._isMarginCall(60_000, 50_000, 0.25)
        self.assertTrue(is_mc)
        self.assertEqual(worth, 5_000)

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
            [61, 60, 63, 62.5],    # Collateral: 1562.5
            [302, 304, 298, 300]    # Collateral: 0
                           ], dtype=np.float32)
        self.assertEqual(broker._get_long_collateral(100_000, security_names, prices), 1562.5)
        self.assertEqual(broker.message.margin_calls, {})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750
        })

        # We will change the prices to make a margin call
        prices = np.array([
            [10.5, 12, 9, 10],    # Collateral: 4250
            [61, 60, 63, 62.5],    # Collateral: 1562.5
            [155, 150, 166, 160]    # Collateral: 1750
                           ], dtype=np.float32)
        self.assertEqual(broker._get_long_collateral(5000, security_names, prices), 5000)
        self.assertEqual(broker.message.margin_calls["long margin call TSLA"].amount, 812.5)
        self.assertEqual(broker._debt_record["long margin call TSLA"], 812.5)
        self.assertEqual(broker.message.margin_calls["long margin call MSFT"].amount, 1750)
        self.assertEqual(broker._debt_record["long margin call MSFT"], 1750)


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
                             datetime(2021, 1, 1), ratio_owned=0.75),    # Collateral: 1562.5
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
                         "long margin call TSLA": MarginCall(2, 1562.5)})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
            "short margin call": 37.5,
            "long margin call TSLA": 1562.5
        })


        # Test with no short margin call
        prices = np.array([
            [102, 104, 98, 100],    # AAPL
            [61, 60, 63, 62.5],     # TSLA
            [302, 304, 298, 300],   # MSFT
            [102, 104, 98, 100],    # V
            [202, 208, 196, 200],   # CAT
            [302, 304, 298, 2298]    # OLN: Collateral - 37262.5
                           ], dtype=np.float32)
        # TODO: Fix bugs in the code to make it pass the test
        broker._update_account_collateral(datetime(2021, 1, 3), security_names, prices)
        self.assertEqual(broker.account.get_cash(), 0)
        self.assertEqual(broker.message.margin_calls, {"long margin call TSLA": MarginCall(2, 1350)})
        self.assertEqual(broker._debt_record, {
            "AAPL": 5000,
            "TSLA": 6250,
            "MSFT": 13750,
            "long margin call TSLA": 1350
        })



        # TODO: Add a test for no margin call at all (Without creating anew broker object)
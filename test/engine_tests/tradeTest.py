from unittest import TestCase
from backtest.engine.trade import Trade, TradeOrder, TradeType
from backtest.engine.trade import BuyLongOrder, SellLongOrder, SellShortOrder, BuyShortOrder
from backtest.engine.trade import SellShort, BuyShort, BuyLong, SellLong
from datetime import datetime


class TestTradeType(TestCase):
    def test_available(self):
        self.assertEqual(TradeType.available(), ["BuyLong", "SellLong", "SellShort", "BuyShort"])

class TestTradeOrder(TestCase):

    def test_init(self):
        # Buy long no margin
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 0,
                           TradeType.BuyLong, datetime(2021, 1, 2))
        self.assertEqual(order.timestamp, datetime(2021, 1, 1))
        self.assertEqual(order.security, "AAPL")
        self.assertEqual(order.security_price_limit, (100, 101))
        self.assertEqual(order.amount, 100)
        self.assertEqual(order.amount_borrowed, 0)
        self.assertEqual(order.trade_type, TradeType.BuyLong)
        self.assertEqual(order.expiry, datetime(2021, 1, 2))
        self.assertEqual(order.margin_trade, False)

        # Now with margin
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 1,
                           TradeType.BuyLong, datetime(2021, 1, 2))
        self.assertEqual(order.timestamp, datetime(2021, 1, 1))
        self.assertEqual(order.security, "AAPL")
        self.assertEqual(order.security_price_limit, (100, 101))
        self.assertEqual(order.amount, 100)
        self.assertEqual(order.amount_borrowed, 1)
        self.assertEqual(order.trade_type, TradeType.BuyLong)
        self.assertEqual(order.expiry, datetime(2021, 1, 2))
        self.assertEqual(order.margin_trade, True)

        # Short selling
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 0, 100,
                           TradeType.SellShort, datetime(2021, 1, 2))
        self.assertEqual(order.timestamp, datetime(2021, 1, 1))
        self.assertEqual(order.security, "AAPL")
        self.assertEqual(order.security_price_limit, (100, 101))
        self.assertEqual(order.amount_borrowed, 100)
        self.assertEqual(order.trade_type, TradeType.SellShort)
        self.assertEqual(order.expiry, datetime(2021, 1, 2))
        self.assertEqual(order.margin_trade, True)
    def test_export(self):
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 0,
                           TradeType.BuyLong, datetime(2021, 1, 2))

        self.assertEqual(order.export(), {
            "type": "TradeOrder.BuyLong",
            "timestamp": "2021-01-01 00:00:00",
            "security": "AAPL",
            "security_price_limit": (100, 101),
            "amount": 100,
            "amount_borrowed": 0,
            "trade_type": "BuyLong",
            "expiry": "2021-01-02 00:00:00",
            "margin_trade": False
        })



    def test_load(self):
        state = {
            "type": "TradeOrder.BuyLong",
            "timestamp": "2021-01-01 00:00:00",
            "security": "AAPL",
            "security_price_limit": (100, 101),
            "amount": 100,
            "amount_borrowed": 0,
            "trade_type": "BuyLong",
            "expiry": "2021-01-02 00:00:00",
            "margin_trade": False
        }
        order = TradeOrder.load(state)
        expected = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 0,
                           TradeType.BuyLong, datetime(2021, 1, 2))
        self.assertEqual(order, expected)

    def test_convert_to_trade(self):
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 105), 100, 0,
                           TradeType.BuyLong, datetime(2021, 1, 5))

        trade = order.convertToTrade(102.45, datetime(2021, 1, 3), "1")
        self.assertEqual(trade.timestamp, datetime(2021, 1, 3))
        self.assertEqual(trade.security, "AAPL")
        self.assertEqual(trade.security_price, 102.45)
        self.assertEqual(trade.amount, 100)
        self.assertEqual(trade.amount_borrowed, 0)
        self.assertEqual(trade.trade_type, TradeType.BuyLong)
        self.assertEqual(trade.margin_trade, False)
        self.assertEqual(trade.transaction_id, "1")
        self.assertEqual(order, trade.trade_order)


class TestTrade(TestCase):
    def test_init(self):
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 105), 50, 50,
                           TradeType.BuyLong, datetime(2021, 1, 5))
        trade = Trade("AAPL", 102.50, 50, 50, "1",
                      datetime(2021, 1, 1), TradeType.BuyLong, order)
        self.assertEqual(trade.timestamp, datetime(2021, 1, 1))
        self.assertEqual(trade.security, "AAPL")
        self.assertEqual(trade.security_price, 102.5)
        self.assertEqual(trade.amount, 50)
        self.assertEqual(trade.amount_borrowed, 50)
        self.assertEqual(trade.trade_type, TradeType.BuyLong)
        self.assertEqual(trade.margin_trade, True)
        self.assertEqual(trade.transaction_id, "1")

    def test_get_cost(self):
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 105), 50, 50,
                           TradeType.BuyLong, datetime(2021, 1, 5))
        trade = Trade("AAPL", 102.50, 50, 50, "1",
                      datetime(2021, 1, 1), TradeType.BuyLong, order)
        self.assertEqual(trade.getCost(), 5125)

    def test_export(self):
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 105), 50, 50,
                           TradeType.BuyLong, datetime(2021, 1, 5))
        trade = Trade("AAPL", 102.50, 50, 50, "1",
                      datetime(2021, 1, 1), TradeType.BuyLong, order)

        expected = {
            "type": f"Trade.BuyLong",
            "security": "AAPL",
            "security_price": 102.5,
            "amount": 50,
            "amount_borrowed": 50,
            "transaction_id": "1",
            "trade_type": "BuyLong",
            "margin_trade": True,
            "timestamp": "2021-01-01 00:00:00",
            "order": order.export()
        }
        state = trade.export()
        self.assertEqual(state, expected)


    def test_load(self):
        state = {
            "type": f"Trade.BuyLong",
            "security": "AAPL",
            "security_price": 102.5,
            "amount": 50,
            "amount_borrowed": 50,
            "transaction_id": "1",
            "trade_type": "BuyLong",
            "margin_trade": True,
            "timestamp": "2021-01-01 00:00:00",
            "order": {
                "type": "TradeOrder.BuyLong",
                "timestamp": "2021-01-01 00:00:00",
                "security": "AAPL",
                "security_price_limit": (100, 105),
                "amount": 50,
                "amount_borrowed": 50,
                "trade_type": "BuyLong",
                "expiry": "2021-01-05 00:00:00",
                "margin_trade": True
            }
        }
        order = TradeOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 105), 50, 50,
                           TradeType.BuyLong, datetime(2021, 1, 5))
        trade = Trade.load(state)
        expected = Trade("AAPL", 102.50, 50, 50, "1",
                      datetime(2021, 1, 1), TradeType.BuyLong, order)
        self.assertEqual(trade, expected)

class TestChildClasses(TestCase):
    def test_orders(self):
        orderBL = BuyLongOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 0,
                               datetime(2021, 1, 2))
        orderSL = SellLongOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 100, 50,
                                 datetime(2021, 1, 2))
        orderSS = SellShortOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 0, 100,
                                 datetime(2021, 1, 2))
        orderBS = BuyShortOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 0, 100,
                                datetime(2021, 1, 2))

        self.assertEqual(orderBL.trade_type, TradeType.BuyLong)
        self.assertEqual(orderBL.margin_trade, False)
        self.assertEqual(orderSL.trade_type, TradeType.SellLong)
        self.assertEqual(orderSL.margin_trade, True)
        self.assertEqual(orderSS.trade_type, TradeType.SellShort)
        self.assertEqual(orderSS.margin_trade, True)
        self.assertEqual(orderBS.trade_type, TradeType.BuyShort)
        self.assertEqual(orderBS.margin_trade, True)

    def test_trades(self):
        orderBL = BuyLongOrder(datetime(2021, 1, 1), "AAPL",
                           (100, 101), 100, 0,
                               datetime(2021, 1, 2))
        orderSL = SellLongOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 100, 50,
                                 datetime(2021, 1, 2))
        orderSS = SellShortOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 0, 100,
                                 datetime(2021, 1, 2))
        orderBS = BuyShortOrder(datetime(2021, 1, 1), "AAPL",
                            (100, 101), 0, 100,
                                datetime(2021, 1, 2))
        tradeBL = BuyLong("AAPL", 102.50, 100, 0, "1",
                      datetime(2021, 1, 1), orderBL)
        tradeSL = SellLong("AAPL", 102.50, 100, 50, "1",
                        datetime(2021, 1, 1), orderSL)
        tradeSS = SellShort("AAPL", 102.50, 0, 100, "1",
                        datetime(2021, 1, 1), orderSS)
        tradeBS = BuyShort("AAPL", 102.50, 0, 100, "1",
                        datetime(2021, 1, 1), orderBS)
        self.assertEqual(tradeBL.trade_type, TradeType.BuyLong)
        self.assertEqual(tradeBL.margin_trade, False)
        self.assertEqual(tradeSL.trade_type, TradeType.SellLong)
        self.assertEqual(tradeSL.margin_trade, True)
        self.assertEqual(tradeSS.trade_type, TradeType.SellShort)
        self.assertEqual(tradeSS.margin_trade, True)
        self.assertEqual(tradeBS.trade_type, TradeType.BuyShort)
        self.assertEqual(tradeBS.margin_trade, True)


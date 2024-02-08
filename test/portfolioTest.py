from unittest import TestCase
from backtest.src.portfolio import Portfolio, Position, TradeStats
from backtest.src.trade import Trade, TradeType, TradeOrder, BuyLong, SellLong, BuyShort, SellShort
from datetime import timedelta, datetime
import json

class TestPosition(TestCase):
    def test_purchase_worth(self):
        # Long
        position = Position("AAPL", 200, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))

        self.assertEqual(position.purchase_worth, 20_000)

        # Short
        self.assertRaises(Exception, Position, "AAPL", 100, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=False)

        position = Position("AAPL", 200, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        self.assertEqual(position.purchase_worth, 20_000)


    def test_dividendGotPaid(self):
        # Long
        position = Position("AAPL", 100, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.time_stock_idx = 10
        self.assertEqual(position.last_dividends_dt, datetime(2021, 1, 1))
        self.assertEqual(position.time_stock_idx, 10)
        position.dividends_got_paid(datetime(2021, 3, 1))
        self.assertEqual(position.last_dividends_dt, datetime(2021, 3, 1))
        self.assertEqual(position.time_stock_idx, 0)

        # Short
        position = Position("AAPL", 200, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        position.time_stock_idx = 42
        self.assertEqual(position.last_dividends_dt, datetime(2021, 1, 1))
        self.assertEqual(position.time_stock_idx, 42)
        position.dividends_got_paid(datetime(2021, 3, 1))
        self.assertEqual(position.last_dividends_dt, datetime(2021, 3, 1))
        self.assertEqual(position.time_stock_idx, 0)

    def test_update_time_stock_idx(self):
        position = Position("AAPL", 200, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.update_time_stock_idx()
        self.assertEqual(position.time_stock_idx, 200)

        position.update_time_stock_idx(3)
        self.assertEqual(position.time_stock_idx, 800)

        # Now, test with change in assets held
        position = Position("AAPL", 200, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.update_time_stock_idx()    # Should be 200
        position.amount = 125
        position.update_time_stock_idx(2)    # Should be 450
        self.assertEqual(position.time_stock_idx, 450)

        position.amount = 50
        position.update_time_stock_idx(3)    # Should be 600
        self.assertEqual(position.time_stock_idx, 600)

        position.amount = 0
        position.update_time_stock_idx()    # Should be 600
        self.assertEqual(position.time_stock_idx, 600)

    def test_add(self):
        # Long position addition
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 100, long=True, average_price=150,
                            average_filled_time=datetime(2021, 2, 1))
        position2.update_time_stock_idx(1)
        position3: Position = position1 + position2
        self.assertEqual(position3.amount, 400)
        self.assertEqual(position3.average_price, 112.5)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 16, 12))
        self.assertEqual(position3.time_stock_idx, 9400)
        self.assertEqual(position3.last_dividends_dt, datetime(2021, 1, 16, 12))
        self.assertTrue(position3.long)

        # Short position addition
        position1 = Position("AAPL", 150, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), margin=True)
        position2.update_time_stock_idx(1)
        position3: Position = position1 + position2
        self.assertEqual(position3.amount, 200)
        self.assertEqual(position3.average_price, 87.5)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 16, 12))
        self.assertEqual(position3.time_stock_idx, 0)
        self.assertFalse(position3.long)

        # Long and short position addition
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), margin=True)
        self.assertRaises(TypeError, position1.__add__, position2)

        # Different stocks
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position2 = Position("MSFT", 50, long=True, average_price=50,
                            average_filled_time=datetime(2021, 2, 1))
        trade = Trade("TSLA", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)

        self.assertRaises(ValueError, position1.__add__, position2)
        self.assertRaises(ValueError, position1.__add__, trade)

        # Position + Trade
        position = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.update_time_stock_idx(31)
        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)

        new_position = position + trade
        self.assertEqual(new_position.amount, 400)
        self.assertEqual(new_position.average_price, 112.5)
        self.assertEqual(new_position.average_filled_time, datetime(2021, 1, 16, 12))
        self.assertEqual(new_position.time_stock_idx, 9300)
        self.assertEqual(new_position.last_dividends_dt, datetime(2021, 1, 16, 12))
        self.assertTrue(new_position.long)

        # Position + trade with wrong trade types
        long = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        trade_sl = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.SellLong, order=None)
        trade_bs = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.BuyShort, order=None)
        trade_ss = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.SellShort, order=None)
        self.assertRaises(ValueError, long.__add__, trade_sl)
        self.assertRaises(ValueError, long.__add__, trade_bs)
        self.assertRaises(ValueError, long.__add__, trade_ss)

        short = Position("AAPL", 150, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        trade_bl = Trade("AAPL", 150, 100, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)
        self.assertRaises(ValueError, short.__add__, trade_bl)
        self.assertRaises(ValueError, short.__add__, trade_bs)
        self.assertRaises(ValueError, short.__add__, trade_sl)

        # Position += Position
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 100, long=True, average_price=150,
                            average_filled_time=datetime(2021, 2, 1))
        position2.update_time_stock_idx(1)
        position1 += position2
        self.assertEqual(position1.amount, 400)
        self.assertEqual(position1.average_price, 112.5)
        self.assertEqual(position1.average_filled_time, datetime(2021, 1, 16, 12))
        self.assertEqual(position1.time_stock_idx, 9400)
        self.assertEqual(position1.last_dividends_dt, datetime(2021, 1, 16, 12))
        self.assertTrue(position1.long)

        # Position += self
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position1.update_time_stock_idx(31)
        position1 += position1
        self.assertEqual(position1.amount, 600)
        self.assertEqual(position1.average_price, 100)
        self.assertEqual(position1.average_filled_time, datetime(2021, 1, 1))
        self.assertEqual(position1.time_stock_idx, 18600)
        self.assertEqual(position1.last_dividends_dt, datetime(2021, 1, 1))
        self.assertTrue(position1.long)



    def test_sub(self):
        # Long position subtraction
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 100, long=True, average_price=150,
                            average_filled_time=datetime(2021, 2, 1))
        position2.update_time_stock_idx(1)
        position3: Position = position1 - position2
        self.assertEqual(position3.amount, 200)
        self.assertEqual(position3.average_price, 100)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 1))
        self.assertEqual(position3.time_stock_idx, 9300)
        self.assertEqual(position3.last_dividends_dt, datetime(2021, 1, 1))
        self.assertTrue(position3.long)

        # Short position subtraction
        position1 = Position("AAPL", 150, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), margin=True)
        position2.update_time_stock_idx(1)
        position3: Position = position1 - position2
        self.assertEqual(position3.amount, 100)
        self.assertEqual(position3.average_price, 100)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 1))
        self.assertEqual(position3.time_stock_idx, 0)
        self.assertFalse(position3.long)

        # Long and short position subtraction
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), margin=True)
        self.assertRaises(TypeError, position1.__sub__, position2)

        # Different stocks
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position2 = Position("MSFT", 50, long=True, average_price=50,
                            average_filled_time=datetime(2021, 2, 1))
        trade = Trade("TSLA", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)
        self.assertRaises(ValueError, position1.__sub__, position2)
        self.assertRaises(ValueError, position1.__sub__, trade)

        # Position - Trade
        position = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.update_time_stock_idx(31)
        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.SellLong, order=None)
        position3 = position - trade
        self.assertEqual(position3.amount, 200)
        self.assertEqual(position3.average_price, 100)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 1))
        self.assertEqual(position3.time_stock_idx, 9300)
        self.assertEqual(position3.last_dividends_dt, datetime(2021, 1, 1))
        self.assertTrue(position3.long)

        # Postion - trade with wrong datetime
        position = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.update_time_stock_idx(31)
        trade = Trade("AAPL", 150, 150, 150, "1",
                      datetime(2019, 3, 1), trade_type=TradeType.BuyLong, order=None)
        self.assertRaises(ValueError, position.__sub__, trade)

        # Position - trade with wrong trade types
        long = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        short = Position("AAPL", 150, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        trade_sl = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.SellLong, order=None)
        trade_bl = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)
        trade_bs = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.BuyShort, order=None)
        trade_ss = Trade("AAPL", 150, 50, 50, "1",
                        datetime(2021, 2, 1), trade_type=TradeType.SellShort, order=None)
        self.assertRaises(ValueError, long.__sub__, trade_bs)
        self.assertRaises(ValueError, long.__sub__, trade_ss)
        self.assertRaises(ValueError, long.__sub__, trade_bl)

        self.assertRaises(ValueError, short.__sub__, trade_bl)
        self.assertRaises(ValueError, short.__sub__, trade_ss)
        self.assertRaises(ValueError, short.__sub__, trade_sl)


    def test_export(self):
        position = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        position.update_time_stock_idx(3)
        self.assertEqual(position.export(), {
            "type": "Position",
            "ticker": "AAPL",
            "amount": 300,
            "long": True,
            "on_margin": True,
            "average_price": 100,
            "average_filled_time": str(datetime(2021, 1, 1)),
            "time_stock_idx": 900,
            "last_dividends_dt": str(datetime(2021, 1, 1))
        })

        # Make sure the dict is JSON serializable
        json.dumps(position.export())

    def test_load(self):
        position_loaded = Position.load({
            "type": "Position",
            "ticker": "AAPL",
            "amount": 300,
            "long": True,
            "on_margin": True,
            "average_price": 100,
            "average_filled_time": str(datetime(2021, 1, 1)),
            "time_stock_idx": 900,
            "last_dividends_dt": str(datetime(2021, 1, 1))
        })
        position_expected = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), margin=True)
        position_expected.update_time_stock_idx(3)

        self.assertEqual(position_expected, position_loaded)


class TestTradeStats(TestCase):
    def test_init(self):
        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)

        self.assertRaises(ValueError, TradeStats, trade, timedelta(weeks=1), 100, 100)

        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.SellLong, order=None)

        # Test no throw
        TradeStats(trade, timedelta(weeks=1), 100, 100)

    def test_export(self):
        order = TradeOrder(datetime(2020, 12, 28), "AAPL", (95, None),
                           50, 50, TradeType.SellLong, datetime(2021, 2, 1))
        trade = order.convertToTrade(100, datetime(2021, 1, 2), "1")
        trade_stats = TradeStats(trade, timedelta(weeks=1), 100, 100)
        self.assertEqual(trade_stats.export(), {
            "type": "TradeStats",
            "trade": trade.export(),
            "duration": 604_800,
            "profit": 100,
            "rel_profit": 100
        })

        # Make sure the dict is JSON serializable
        json.dumps(trade_stats.export())


    def test_load(self):
        order = TradeOrder(datetime(2020, 12, 28), "AAPL", (95, None),
                           50, 50, TradeType.SellLong, datetime(2021, 2, 1))
        trade = order.convertToTrade(100, datetime(2021, 1, 2), "1")
        expected_trade_stats = TradeStats(trade, timedelta(weeks=1), 100, 100)
        trade_stats = TradeStats.load({
            "type": "TradeStats",
            "trade": trade.export(),
            "duration": 604_800,
            "profit": 100,
            "rel_profit": 100
        })

        self.assertEqual(expected_trade_stats, trade_stats)



class TestPortfolio(TestCase):
    def test_trade(self):
        # ------------------------------
        # Buy Long
        # ------------------------------
        portfolio = Portfolio(6.99, False)

        # No positions yet
        trade1 = BuyLong("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), order=None)
        portfolio.trade(trade1)
        self.assertEqual(portfolio._long, {"AAPL": Position("AAPL", 100,
                                                        True, 150, datetime(2021, 2, 1), margin=True)})

        # With existing position
        trade2 = BuyLong("AAPL", 100, 50, 50, "2",
                      datetime(2021, 2, 1), order=None)
        portfolio.trade(trade2)
        self.assertEqual(portfolio._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125,
                                                            datetime(2021, 2, 1), margin=True)})

        # With new position
        trade3 = BuyLong("MSFT", 100, 50, 50, "3",
                      datetime(2021, 2, 1), order=None)
        portfolio.trade(trade3)
        self.assertEqual(portfolio._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125, datetime(2021, 2, 1),
                                                            margin=True),
                                        "MSFT": Position("MSFT", 100, True,
                                                         100, datetime(2021, 2, 1), margin=True)})

        self.assertEqual(portfolio._trades, [trade1, trade2, trade3])

        # ------------------------------
        # Sell Long
        # ------------------------------
        debt_record = {}
        portfolio = Portfolio(6.99, True, debt_record)
        trade1 = BuyLong("AAPL", 100, 250, 250, "1",
                      datetime(2021, 2, 1), order=None)
        portfolio.trade(trade1)
        # ------------------------------
        # This is normally the job of the broker
        if trade1.security in debt_record:
            debt_record[trade1.security] += trade1.amount_borrowed * 100 * 1.01
        else:
            debt_record[trade1.security] = trade1.amount_borrowed * 100 * 1.01
        # ------------------------------

        # No positions yet
        self.assertRaises(RuntimeError, portfolio.trade, SellLong("MSFT", 100, 50,
                                                                  50, "2",
                                                                  datetime(2021, 2, 1), order=None))

        # Insufficient amount
        self.assertRaises(RuntimeError, portfolio.trade, SellLong("AAPL", 100, 500,
                                                                  500, "2",
                                                                  datetime(2021, 2, 1), order=None))

        # With existing position
        trade2 = SellLong("AAPL", 150, 300, 0, "3",
                        datetime(2021, 2, 1), order=None)
        long, short = portfolio["AAPL"]
        print(long)
        portfolio.trade(trade2)
        self.assertEqual(portfolio._long, {"AAPL": Position("AAPL", 100,
                                                        True, 100, datetime(2021, 2, 1), margin=True)})

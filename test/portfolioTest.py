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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), ratio_owned=0.)
        position2.update_time_stock_idx(1)
        position3: Position = position1 + position2
        self.assertEqual(position3.amount, 200)
        self.assertEqual(position3.average_price, 87.5)
        self.assertEqual(position3.average_filled_time, datetime(2021, 1, 16, 12))
        self.assertEqual(position3.time_stock_idx, 0)
        self.assertFalse(position3.long)

        # Long and short position addition
        position1 = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.5)
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
        position1.update_time_stock_idx(31)
        position2 = Position("AAPL", 50, long=False, average_price=50,
                            average_filled_time=datetime(2021, 2, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 2, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.)
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
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.5)
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
            "last_dividends_dt": str(datetime(2021, 1, 1)),
            "ratio_owned": 0.5
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
            "last_dividends_dt": str(datetime(2021, 1, 1)),
            "ratio_owned": 0.5
        })
        position_expected = Position("AAPL", 300, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1), ratio_owned=0.5)
        position_expected.update_time_stock_idx(3)

        self.assertEqual(position_expected, position_loaded)


class TestTradeStats(TestCase):
    def test_init(self):
        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.BuyLong, order=None)

        self.assertRaises(ValueError, TradeStats, trade, timedelta(weeks=1), 100, 100, 0.5)

        trade = Trade("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), trade_type=TradeType.SellLong, order=None)

        # Test no throw
        TradeStats(trade, timedelta(weeks=1), 100, 100, 0.5)

    def test_export(self):
        order = TradeOrder(datetime(2020, 12, 28), "AAPL", (95, None),
                           50, 50, TradeType.SellLong, datetime(2021, 2, 1))
        trade = order.convertToTrade(100, datetime(2021, 1, 2), "1")
        trade_stats = TradeStats(trade, timedelta(weeks=1), 100, 100, ratio_owned=0.5)
        self.assertEqual(trade_stats.export(), {
            "type": "TradeStats",
            "trade": trade.export(),
            "duration": 604_800,
            "profit": 100,
            "rel_profit": 100,
            "ratio_owned": 0.5
        })

        # Make sure the dict is JSON serializable
        json.dumps(trade_stats.export())


    def test_load(self):
        order = TradeOrder(datetime(2020, 12, 28), "AAPL", (95, None),
                           50, 50, TradeType.SellLong, datetime(2021, 2, 1))
        trade = order.convertToTrade(100, datetime(2021, 1, 2), "1")
        expected_trade_stats = TradeStats(trade, timedelta(weeks=1), 100, 100, ratio_owned=0.5)
        trade_stats = TradeStats.load({
            "type": "TradeStats",
            "trade": trade.export(),
            "duration": 604_800,
            "profit": 100,
            "rel_profit": 100,
            "ratio_owned": 0.5
        })

        self.assertEqual(expected_trade_stats, trade_stats)



class TestPortfolio(TestCase):
    def test_trade_BL(self):
        # ------------------------------
        # Buy Long
        # ------------------------------
        portfolio_abs = Portfolio(6.99, False)
        portfolio_rel = Portfolio(1, True)     # 1% cost

        # Add trade to portfolio - New position
        trade1 = BuyLong("AAPL", 150, 50, 50, "1",
                      datetime(2021, 2, 1), order=None)
        total_abs = portfolio_abs.trade(trade1)
        total_rel = portfolio_rel.trade(trade1)
        # Verify if the worth is accurately computed
        self.assertEqual(total_abs, -7506.99)
        self.assertEqual(total_rel, -7575)
        # Verify that the portfolio is correctly updated
        self.assertEqual(portfolio_abs._long, {"AAPL": Position("AAPL", 100,
                                                        True, 150, datetime(2021, 2, 1), 0.5)})
        self.assertEqual(portfolio_rel._long, {"AAPL": Position("AAPL", 100,
                                                        True, 150, datetime(2021, 2, 1), 0.5)})
        # Verify that the debt record is correctly updated
        self.assertEqual(portfolio_abs._debt_record, {"AAPL": 7500})
        self.assertEqual(portfolio_rel._debt_record, {"AAPL": 7575})

        # Add trade to portfolio - Existing position
        trade2 = BuyLong("AAPL", 100, 50, 50, "2",
                      datetime(2021, 2, 1), order=None)
        portfolio_abs.trade(trade2)
        portfolio_rel.trade(trade2)
        self.assertEqual(portfolio_abs._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125,
                                                            datetime(2021, 2, 1), 0.5)})
        self.assertEqual(portfolio_rel._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125,
                                                            datetime(2021, 2, 1), 0.5)})
        self.assertEqual(portfolio_abs._debt_record, {"AAPL": 12500})
        self.assertEqual(portfolio_rel._debt_record, {"AAPL": 12625})

        # With new position
        trade3 = BuyLong("MSFT", 100, 50, 50, "3",
                      datetime(2021, 2, 1), order=None)
        total_abs = portfolio_abs.trade(trade3)
        total_rel = portfolio_rel.trade(trade3)
        self.assertEqual(total_abs, -5006.99)
        self.assertEqual(total_rel, -5050)
        self.assertEqual(portfolio_abs._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125, datetime(2021, 2, 1),
                                                            0.5),
                                        "MSFT": Position("MSFT", 100, True,
                                                         100, datetime(2021, 2, 1), 0.5)})
        self.assertEqual(portfolio_rel._long, {"AAPL": Position("AAPL", 200,
                                                        True, 125, datetime(2021, 2, 1),
                                                            0.5),
                                        "MSFT": Position("MSFT", 100, True,
                                                         100, datetime(2021, 2, 1), 0.5)})

        # Verify that the historical trades are saved.
        self.assertEqual(portfolio_abs._trades, [trade1, trade2, trade3])
        self.assertEqual(portfolio_rel._trades, [trade1, trade2, trade3])
        self.assertEqual(portfolio_abs._debt_record, {"AAPL": 12500, "MSFT": 5000})
        self.assertEqual(portfolio_rel._debt_record, {"AAPL": 12625, "MSFT": 5050})

    def test_trade_SL(self):
        # ------------------------------
        # Sell Long
        # ------------------------------
        portfolio_abs = Portfolio(6.99, False)
        portfolio_rel = Portfolio(1, True)     # 1% cost
        # Add shares in portfolio
        trade1 = BuyLong("AAPL", 100, 250, 250, "1",
                         datetime(2021, 2, 1), order=None)
        portfolio_abs.trade(trade1)
        portfolio_rel.trade(trade1)
        # Now, the portfolio should have 500 shares of AAPL.  Where 50% is bought on margin.
        # Debt record:
        # portfolio_abs._debt_record = {"AAPL": 25000}
        # portfolio_rel._debt_record = {"AAPL": 25250}


        # Test to sell if no open positions
        trade2 = SellLong("MSFT", 100, 50, 50, "2",
                          datetime(2021, 2, 1), order=None)
        self.assertRaises(RuntimeError, portfolio_abs.trade, trade2)
        self.assertRaises(RuntimeError, portfolio_rel.trade, trade2)

        # Test to sell if insufficient amount
        trade2 = SellLong("AAPL", 150, 500, 500, "2",
                          datetime(2021, 2, 1), order=None)
        self.assertRaises(RuntimeError, portfolio_abs.trade, trade2)
        self.assertRaises(RuntimeError, portfolio_rel.trade, trade2)

        # Test to sell with existing position
        trade2 = SellLong("AAPL", 150, 75, 0, "3",
                          datetime(2021, 2, 15), order=None)
        total_abs = portfolio_abs.trade(trade2)
        total_rel = portfolio_rel.trade(trade2)

        # Test if debt record is accurately computed and updated
        # relative = 75 / 500 = 0.15
        # abs: 0.15 * 25000 = 3750$
        # rel: 0.15 * 25250 = 3787.5$
        self.assertAlmostEqual(portfolio_abs._debt_record["AAPL"], 21250)
        self.assertAlmostEqual(portfolio_rel._debt_record["AAPL"], 21462.5)

        # Test Calculation of worth of the trade
        self.assertEqual(total_abs, 7493.01)
        self.assertEqual(total_rel, 7350)

        # Test trade stats
        abs_stat = TradeStats(trade2, timedelta(weeks=2), 3736.02, 99.44184041, 0.5)
        self.assertEqual(portfolio_abs._trades[1].trade, abs_stat.trade)
        self.assertEqual(portfolio_abs._trades[1].duration, abs_stat.duration)
        self.assertAlmostEqual(portfolio_abs._trades[1].profit, abs_stat.profit)
        self.assertAlmostEqual(portfolio_abs._trades[1].rel_profit, abs_stat.rel_profit)
        self.assertEqual(portfolio_abs._trades[1].ratio_owned, abs_stat.ratio_owned)

        rel_stat = TradeStats(trade2, timedelta(weeks=2), 3562.5, 94.05940594, 0.5)
        self.assertEqual(portfolio_rel._trades[1].trade, rel_stat.trade)
        self.assertEqual(portfolio_rel._trades[1].duration, rel_stat.duration)
        self.assertAlmostEqual(portfolio_rel._trades[1].profit, rel_stat.profit)
        self.assertAlmostEqual(portfolio_rel._trades[1].rel_profit, rel_stat.rel_profit)
        self.assertEqual(portfolio_rel._trades[1].ratio_owned, rel_stat.ratio_owned)

        # Test if the position is correctly updated
        self.assertEqual(portfolio_abs._long, {"AAPL": Position("AAPL", 425, True, 100,
                                                        datetime(2021, 2, 1), 0.5)})
        self.assertEqual(portfolio_rel._long, {"AAPL": Position("AAPL", 425, True, 100,
                                                        datetime(2021, 2, 1), 0.5)})


        # Now, we will make a new test with no margin
        portfolio_abs = Portfolio(9.99, False)
        portfolio_rel = Portfolio(0.5, True)     # 0.5% cost
        # Add shares in portfolio
        trade1 = BuyLong("AAPL", 100, 500, 0, "1",
                         datetime(2021, 2, 1), order=None)
        portfolio_abs.trade(trade1)
        portfolio_rel.trade(trade1)
        # Now, the portfolio should have 500 shares of AAPL.
        # Debt record:
        # portfolio_abs._debt_record = {"AAPL": 0}
        # portfolio_rel._debt_record = {"AAPL": 0}

        # Test to sell with existing position
        trade2 = SellLong("AAPL", 150, 100, 0, "3",
                          datetime(2021, 2, 15), order=None)
        total_abs = portfolio_abs.trade(trade2)
        total_rel = portfolio_rel.trade(trade2)

        # Test if debt record is accurately computed and updated
        # abs: 0
        # rel: 0
        self.assertEqual(portfolio_abs._debt_record, {"AAPL": 0})
        self.assertEqual(portfolio_rel._debt_record, {"AAPL": 0})

        # Test Calculation of worth of the trade
        self.assertEqual(total_abs, 14990.01)
        self.assertAlmostEqual(total_rel, 14925)

        # Test trade stats
        abs_stat = TradeStats(trade2, timedelta(weeks=2), 4980.02, 49.75049925, 1.)
        self.assertEqual(portfolio_abs._trades[1].trade, abs_stat.trade)
        self.assertEqual(portfolio_abs._trades[1].duration, abs_stat.duration)
        self.assertAlmostEqual(portfolio_abs._trades[1].profit, abs_stat.profit)
        self.assertAlmostEqual(portfolio_abs._trades[1].rel_profit, abs_stat.rel_profit)
        self.assertEqual(portfolio_abs._trades[1].ratio_owned, abs_stat.ratio_owned)

        rel_stat = TradeStats(trade2, timedelta(weeks=2), 4875, 48.50746269, 1.)
        self.assertEqual(portfolio_rel._trades[1].trade, rel_stat.trade)
        self.assertEqual(portfolio_rel._trades[1].duration, rel_stat.duration)
        self.assertAlmostEqual(portfolio_rel._trades[1].profit, rel_stat.profit)
        self.assertAlmostEqual(portfolio_rel._trades[1].rel_profit, rel_stat.rel_profit)
        self.assertEqual(portfolio_rel._trades[1].ratio_owned, rel_stat.ratio_owned)


        # Test trade stats with small margin (17%)
        portfolio_abs = Portfolio(6.99, False)
        portfolio_rel = Portfolio(1, True)     # 1% cost
        # Add shares in portfolio
        trade1 = BuyLong("AAPL", 100, 83, 17, "1",
                         datetime(2021, 2, 1), order=None)
        portfolio_abs.trade(trade1)
        portfolio_rel.trade(trade1)

        # We have 100 shares of AAPL with 17% margin
        # Debt record:
        # portfolio_abs._debt_record = {"AAPL": 1700}
        # portfolio_rel._debt_record = {"AAPL": 1717}

        # Test to sell with existing position
        trade2 = SellLong("AAPL", 150, 50, 0, "3",
                          datetime(2021, 2, 15), order=None)
        total_abs = portfolio_abs.trade(trade2)
        total_rel = portfolio_rel.trade(trade2)

        # Test if total is accurately computed
        self.assertEqual(total_abs, 6643.01)    # Paying 850$ of debt
        self.assertEqual(total_rel, 6566.5)       # Paying 858.5$ of debt

        # Test Stats
        abs_stat = TradeStats(trade2, timedelta(weeks=2), 2486.02, 59.80336734, 0.83)
        self.assertEqual(portfolio_abs._trades[1].trade, abs_stat.trade)
        self.assertEqual(portfolio_abs._trades[1].duration, abs_stat.duration)
        self.assertAlmostEqual(portfolio_abs._trades[1].profit, abs_stat.profit)
        self.assertAlmostEqual(portfolio_abs._trades[1].rel_profit, abs_stat.rel_profit)
        self.assertEqual(portfolio_abs._trades[1].ratio_owned, abs_stat.ratio_owned)

        rel_stat = TradeStats(trade2, timedelta(weeks=2), 2375, 56.66229274, 0.83)
        self.assertEqual(portfolio_rel._trades[1].trade, rel_stat.trade)
        self.assertEqual(portfolio_rel._trades[1].duration, rel_stat.duration)
        self.assertAlmostEqual(portfolio_rel._trades[1].profit, rel_stat.profit)
        self.assertAlmostEqual(portfolio_rel._trades[1].rel_profit, rel_stat.rel_profit)


    def test_trade_SL(self):
        # ------------------------------
        # Sell Short
        # ------------------------------
        portfolio_abs = Portfolio(6.99, False)
        portfolio_rel = Portfolio(1, True)     # 1% cost

        # Test sell short
        trade1 = SellShort("AAPL", 100, 0, 100, "1",
                           datetime(2021, 2, 1), order=None)
        total_abs = portfolio_abs.trade(trade1)
        total_rel = portfolio_rel.trade(trade1)

        # Test if the worth is accurately computed
        self.assertEqual(total_abs, 9993.01)
        self.assertEqual(total_rel, 9900)

        # Test if the portfolio is correctly updated
        self.assertEqual(portfolio_abs._short, {"AAPL": Position("AAPL", 100, False, 100,
                                                         datetime(2021, 2, 1), 0.)})
        self.assertEqual(portfolio_rel._short, {"AAPL": Position("AAPL", 100, False, 100,
                                                            datetime(2021, 2, 1), 0.)})

        trade2 = SellShort("MSFT", 100, 0, 100, "2",
                           datetime(2021, 2, 1), order=None)
        total_abs = portfolio_abs.trade(trade2)
        total_rel = portfolio_rel.trade(trade2)

        # Test if the historical trades are saved.
        self.assertEqual(portfolio_abs._trades, [trade1, trade2])
        self.assertEqual(portfolio_rel._trades, [trade1, trade2])

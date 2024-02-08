from unittest import TestCase
from backtest.src.portfolio import Portfolio, Position, TradeStats
from datetime import timedelta, datetime


class TestPosition(TestCase):
    def test_purchase_worth(self):
        # Long
        position = Position("AAPL", 100, 100, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))

        self.assertEqual(position.purchase_worth, 20_000)

        # Short
        self.assertRaises(Exception, Position, "AAPL", 100, 100, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))

        position = Position("AAPL", 0, 200, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        self.assertEqual(position.purchase_worth, 20_000)


    def  test_getTotal(self):
        # Long
        position = Position("AAPL", 100, 100, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        self.assertEqual(position.get_total(), 200)

        # Short
        position = Position("AAPL", 0, 200, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        self.assertEqual(position.get_total(), 200)


    def test_dividendGotPaid(self):
        # Long
        position = Position("AAPL", 100, 100, long=True, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.time_stock_idx = 10
        self.assertEqual(position.last_dividends_dt, datetime(2021, 1, 1))
        self.assertEqual(position.time_stock_idx, 10)
        position.dividends_got_paid(datetime(2021, 3, 1))
        self.assertEqual(position.last_dividends_dt, datetime(2021, 3, 1))
        self.assertEqual(position.time_stock_idx, 0)

        # Short
        position = Position("AAPL", 0, 200, long=False, average_price=100,
                            average_filled_time=datetime(2021, 1, 1))
        position.time_stock_idx = 42
        self.assertEqual(position.last_dividends_dt, datetime(2021, 1, 1))
        self.assertEqual(position.time_stock_idx, 42)
        position.dividends_got_paid(datetime(2021, 3, 1))
        self.assertEqual(position.last_dividends_dt, datetime(2021, 3, 1))
        self.assertEqual(position.time_stock_idx, 0)
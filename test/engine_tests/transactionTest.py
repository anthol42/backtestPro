from unittest import TestCase
from backtest.engine.transaction import Transaction, TransactionType
from datetime import datetime

class TestTransactionType(TestCase):
    def test_str(self):
        self.assertEqual(str(TransactionType.DEPOSIT), 'DEPOSIT')
        self.assertEqual(str(TransactionType.WITHDRAWAL), 'WITHDRAWAL')

    def test_from_str(self):
        self.assertEqual(TransactionType.from_str('DEPOSIT'), TransactionType.DEPOSIT)
        self.assertEqual(TransactionType.from_str('WITHDRAWAL'), TransactionType.WITHDRAWAL)
        with self.assertRaises(ValueError):
            TransactionType.from_str('INVALID')

    def test_bool_op(self):
        depo = TransactionType.DEPOSIT
        self.assertEqual(depo, TransactionType.DEPOSIT)
        self.assertNotEqual(depo, TransactionType.WITHDRAWAL)

        withdraw = TransactionType.WITHDRAWAL
        self.assertEqual(withdraw, TransactionType.WITHDRAWAL)
        self.assertNotEqual(withdraw, TransactionType.DEPOSIT)

class TestTransaction(TestCase):
    t1 = Transaction(10.427, TransactionType.DEPOSIT, datetime(2021, 10, 28, hour=6, minute=0))
    t2 = Transaction(42.00, TransactionType.WITHDRAWAL, datetime(2021, 10, 17, hour=7, minute=42),
                     comment='This is a comment', transaction_id='1234567890')
    def test_str(self):
        self.assertEqual(str(self.t1), 'DEPOSIT 10.43$, 2021-10-28 06:00:00')
        self.assertEqual(str(self.t2), 'WITHDRAWAL 42.0$, 2021-10-17 07:42:00, comment=\'This is a comment\'')

    def test_export(self):
        t1_exp = {
            "type": "Transaction.DEPOSIT",
            "amount": 10.427,
            "dt": "2021-10-28 06:00:00",
            "transaction_id": None,
            "comment": None,
            "t": "DEPOSIT"
        }
        self.assertEqual(self.t1.export(), t1_exp)

        t2_exp = {
            "type": "Transaction.WITHDRAWAL",
            "amount": 42.0,
            "dt": "2021-10-17 07:42:00",
            "transaction_id": "1234567890",
            "comment": "This is a comment",
            "t": "WITHDRAWAL"
        }

        self.assertEqual(self.t2.export(), t2_exp)

    def test_load(self):
        exp1 = {
            "type": "Transaction.DEPOSIT",
            "amount": 10.427,
            "dt": "2021-10-28 06:00:00",
            "transaction_id": None,
            "comment": None,
            "t": "DEPOSIT"
        }

        self.assertEqual(Transaction.load(exp1), self.t1)

        exp2 = {
            "type": "Transaction.WITHDRAWAL",
            "amount": 42.0,
            "dt": "2021-10-17 07:42:00",
            "transaction_id": "1234567890",
            "comment": "This is a comment",
            "t": "WITHDRAWAL"
        }
        self.assertEqual(Transaction.load(exp2), self.t2)
from unittest import TestCase
from src.backtest.engine import Account, CollateralUpdate, TransactionType, Transaction, CollateralUpdateType
from datetime import datetime

class TestCollateralUpdate(TestCase):
    update1 = CollateralUpdate(100, datetime(2021, 1, 28, 10, 45), "Test", CollateralUpdateType.UPDATE)
    update2 = CollateralUpdate(200_000, datetime(2019, 12, 25, 0, 0), "Christmas", CollateralUpdateType.ADD)
    exp1 = {
        "type": "CollateralUpdate",
        "amount": 100,
        "dt": "2021-01-28 10:45:00",
        "message": "Test",
        "collateral_update_type": "UPDATE"
    }
    exp2 = {
        "type": "CollateralUpdate",
        "amount": 200_000,
        "dt": "2019-12-25 00:00:00",
        "message": "Christmas",
        "collateral_update_type": "ADD"
    }
    def test_export(self):

        self.assertEqual(self.update1.export(), self.exp1)

        self.assertEqual(self.update2.export(), self.exp2)


    def test_load(self):
        self.assertEqual(CollateralUpdate.load(self.exp1), self.update1)
        self.assertEqual(CollateralUpdate.load(self.exp2), self.update2)


class TestAccount(TestCase):

    def assertNoRaise(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")
    def test_deposit(self):
        account1 = Account(10_000, False)
        account1.deposit(100, datetime(2021, 1, 28, 10, 45), "1", "Test")
        self.assertEqual(account1._cash, 10_100)
        self.assertEqual(account1._transactions[0].amount, 100)
        self.assertEqual(account1._transactions[0].t, TransactionType.DEPOSIT)
        self.assertEqual(account1._transactions[0].dt, datetime(2021, 1, 28, 10, 45))
        self.assertEqual(account1._transactions[0].transaction_id, "1")
        self.assertEqual(account1._transactions[0].comment, "Test")
        self.assertEqual(account1._account_worth, [10_000, 10_100])
        self.assertEqual(account1._account_worth, [10_000, 10_100])
        self.assertEqual(account1.n, 1)
        self.assertRaises(RuntimeError, account1.deposit, 100, datetime(2021, 1, 28, 10, 45), "1", "Test")

    def test_withdrawal(self):
        account1 = Account(10_000, False)
        account2 = Account(10_000, True)
        account1.withdrawal(100, datetime(2021, 1, 29, 15, 30), "2", "Test")
        self.assertEqual(account1._cash, 9_900)
        self.assertEqual(account1._transactions[0].amount, 100)
        self.assertEqual(account1._transactions[0].t, TransactionType.WITHDRAWAL)
        self.assertEqual(account1._transactions[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account1._transactions[0].transaction_id, "2")
        self.assertEqual(account1._transactions[0].comment, "Test")
        self.assertEqual(account1._account_worth, [10_000, 9_900])
        self.assertEqual(account1.n, 1)
        self.assertRaises(RuntimeError, account1.withdrawal, 100, datetime(2021, 1, 29, 15, 30), "2", "Test")

        # Test if I can withdraw more than I have
        self.assertNoRaise(account1.withdrawal, 9_900, datetime(2021, 1, 29, 15, 30), "3", "Test")
        self.assertEqual(account1.n, 2)
        self.assertRaises(RuntimeError, account1.withdrawal, 1, datetime(2021, 1, 29, 15, 30), "4", "Test")

        # If allow_debt is True, I should be able to withdraw more than I have
        self.assertNoRaise(account2.withdrawal, 11_000, datetime(2021, 1, 29, 15, 30), "2", "Test")
        self.assertEqual(account2.n, 1)
        self.assertEqual(account2._cash, -1_000)
        self.assertEqual(account2._transactions[0].amount, 11_000)
        self.assertEqual(account2._transactions[0].t, TransactionType.WITHDRAWAL)
        self.assertEqual(account2._transactions[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account2._transactions[0].transaction_id, "2")

    def test_update_collateral(self):
        account1 = Account(10_000, False)
        account2 = Account(10_000, True)
        account1.update_collateral(10_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account1._collateral, 10_000)
        self.assertEqual(account1._collateral_history[0].amount, 10_000)
        self.assertEqual(account1._collateral_history[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account1._collateral_history[0].message, "[UPDATE] - Test")
        self.assertRaises(RuntimeError, account1.update_collateral, 10_001, datetime(2021, 1, 29, 15, 30), "Test")

        # If allow_debt is True, I should be able to update collateral to a value higher than the current cash
        self.assertNoRaise(account2.update_collateral, 11_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account2._collateral, 11_000)
        self.assertEqual(account2._collateral_history[0].amount, 11_000)
        self.assertEqual(account2._collateral_history[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account2._collateral_history[0].message, "[UPDATE] - Test")


    def test_add_collateral(self):
        account1 = Account(10_000, False)
        account1.add_collateral(10_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account1._collateral, 10_000)
        self.assertEqual(account1._collateral_history[0].amount, 10_000)
        self.assertEqual(account1._collateral_history[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account1._collateral_history[0].message, "[ADD] - Test")
        self.assertRaises(RuntimeError, account1.add_collateral, 1, datetime(2021, 1, 29, 15, 30), "Test")

        # If allow_debt is True, I should be able to add collateral to a value higher than the current cash
        account2 = Account(10_000, True)
        self.assertNoRaise(account2.add_collateral, 11_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account2._collateral, 11_000)
        self.assertEqual(account2._collateral_history[0].amount, 11_000)
        self.assertEqual(account2._collateral_history[0].dt, datetime(2021, 1, 29, 15, 30))
        self.assertEqual(account2._collateral_history[0].message, "[ADD] - Test")

    def test_remove_collateral(self):
        account1 = Account(10_000, False)
        account1.add_collateral(10_000, datetime(2021, 1, 28, 15, 30), "Test")
        account1.remove_collateral(5_000, datetime(2021, 1, 29, 15, 45), "Test")
        self.assertEqual(account1._collateral, 5_000)
        self.assertEqual(account1._collateral_history[1].amount, 5_000)
        self.assertEqual(account1._collateral_history[1].dt, datetime(2021, 1, 29, 15, 45))
        self.assertEqual(account1._collateral_history[1].message, "[REMOVE] - Test")
        self.assertEqual(account1.get_cash(), 5_000)
        self.assertRaises(RuntimeError, account1.remove_collateral, 5_001, datetime(2021, 1, 29, 15, 45), "Test")


    def test_get_cash(self):
        accout1 = Account(10_000, False)
        self.assertEqual(accout1.get_cash(), 10_000)

        accout1.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(accout1.get_cash(), 5_000)

        accout1.add_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(accout1.get_cash(), 0)

        accout2 = Account(10_000, True)
        self.assertEqual(accout2.get_cash(), 10_000)
        accout2.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(accout2.get_cash(), 5_000)
        accout2.add_collateral(6_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(accout2.get_cash(), -1_000)

    def test_get_total_cash(self):
        account1 = Account(10_000, False)
        self.assertEqual(account1.get_total_cash(), 10_000)
        account1.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account1.get_total_cash(), 10_000)
        account1.add_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account1.get_total_cash(), 10_000)

        account2 = Account(10_000, True)
        self.assertEqual(account2.get_total_cash(), 10_000)
        account2.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account2.get_total_cash(), 10_000)
        account2.add_collateral(6_000, datetime(2021, 1, 29, 15, 30), "Test")
        self.assertEqual(account2.get_total_cash(), 10_000)

    def test_get_state(self):
        account1 = Account(10_000, False)
        account1.deposit(100, datetime(2021, 1, 28, 10, 45), "1", "Test")
        account1.withdrawal(200, datetime(2021, 1, 29, 15, 30), "2", "Test")
        account1.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        account1.deposit(200, datetime(2021, 1, 28, 10, 45), "3", "Test")
        account1.withdrawal(100, datetime(2021, 1, 29, 15, 30), "4", "Test")
        account1.add_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        account1_state = account1.get_state()
        account1_state["previous_ids"] = sorted(account1_state["previous_ids"])
        self.assertEqual(account1_state, {
            "type": "Account",
            "cash": 10_000,
            "collateral": 10_000,
            "transactions": [
                {"type": f"Transaction.DEPOSIT", "amount": 100, "dt": "2021-01-28 10:45:00", "transaction_id": "1", "comment": "Test", "t": "DEPOSIT"},
                {"type": f"Transaction.WITHDRAWAL", "amount": 200, "dt": "2021-01-29 15:30:00", "transaction_id": "2", "comment": "Test", "t": "WITHDRAWAL"},
                {"type": f"Transaction.DEPOSIT", "amount": 200, "dt": "2021-01-28 10:45:00", "transaction_id": "3", "comment": "Test", "t": "DEPOSIT"},
                {"type": f"Transaction.WITHDRAWAL", "amount": 100, "dt": "2021-01-29 15:30:00", "transaction_id": "4", "comment": "Test", "t": "WITHDRAWAL"}
            ],
            "account_worth": [10_000, 10_100, 9_900, 10_100, 10_000],
            "allow_debt": False,
            "collateral_history": [
                {"type": "CollateralUpdate", "amount": 5_000, "dt": "2021-01-29 15:30:00", "message": "[UPDATE] - Test", "collateral_update_type": "UPDATE"},
                {"type": "CollateralUpdate", "amount": 5_000, "dt": "2021-01-29 15:30:00", "message": "[ADD] - Test", "collateral_update_type": "ADD"}
            ],
            "previous_ids": ["1", "2", "3", "4"],
        })

    def test_stats(self):
        account1 = Account(10_000, False)
        account1.deposit(100, datetime(2021, 1, 28, 10, 45), "1", "Test")
        account1.withdrawal(200, datetime(2021, 1, 29, 15, 30), "2", "Test")
        account1.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        account1.deposit(200, datetime(2021, 1, 28, 10, 45), "3", "Test")
        account1.withdrawal(100, datetime(2021, 1, 29, 15, 30), "4", "Test")
        account1.add_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")

        self.assertEqual(account1.stats(), {
            "transactions": [Transaction(100, TransactionType.DEPOSIT, datetime(2021, 1, 28, 10, 45), "1", "Test"),
                             Transaction(200, TransactionType.WITHDRAWAL, datetime(2021, 1, 29, 15, 30), "2", "Test"),
                             Transaction(200, TransactionType.DEPOSIT, datetime(2021, 1, 28, 10, 45), "3", "Test"),
                             Transaction(100, TransactionType.WITHDRAWAL, datetime(2021, 1, 29, 15, 30), "4", "Test")],
            "cash_curve": [10000, 10100, 9900, 10100, 10000],
            "final_cash": 10000
        })

    def test_load_state(self):
        state = {
            "type": "Account",
            "cash": 10_000,
            "collateral": 10_000,
            "transactions": [
                {"type": f"Transaction.DEPOSIT", "amount": 100, "dt": "2021-01-28 10:45:00", "transaction_id": "1", "comment": "Test", "t": "DEPOSIT"},
                {"type": f"Transaction.WITHDRAWAL", "amount": 200, "dt": "2021-01-29 15:30:00", "transaction_id": "2", "comment": "Test", "t": "WITHDRAWAL"},
                {"type": f"Transaction.DEPOSIT", "amount": 200, "dt": "2021-01-28 10:45:00", "transaction_id": "3", "comment": "Test", "t": "DEPOSIT"},
                {"type": f"Transaction.WITHDRAWAL", "amount": 100, "dt": "2021-01-29 15:30:00", "transaction_id": "4", "comment": "Test", "t": "WITHDRAWAL"}
            ],
            "account_worth": [10_000, 10_100, 9_900, 10_100, 10_000],
            "allow_debt": False,
            "collateral_history": [
                {"type": "CollateralUpdate", "amount": 5_000, "dt": "2021-01-29 15:30:00", "message": "[UPDATE] - Test", "collateral_update_type": "UPDATE"},
                {"type": "CollateralUpdate", "amount": 5_000, "dt": "2021-01-29 15:30:00", "message": "[ADD] - Test", "collateral_update_type": "ADD"}
            ],
            "previous_ids": ["1", "2", "3", "4"],
        }
        account1 = Account(10_000, False)
        account1.deposit(100, datetime(2021, 1, 28, 10, 45), "1", "Test")
        account1.withdrawal(200, datetime(2021, 1, 29, 15, 30), "2", "Test")
        account1.update_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")
        account1.deposit(200, datetime(2021, 1, 28, 10, 45), "3", "Test")
        account1.withdrawal(100, datetime(2021, 1, 29, 15, 30), "4", "Test")
        account1.add_collateral(5_000, datetime(2021, 1, 29, 15, 30), "Test")

        self.assertTrue(Account.load_state(state) == account1)
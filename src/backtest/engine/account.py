from .transaction import Transaction, TransactionType
from datetime import datetime
from typing import List, Optional
from enum import Enum

class CollateralUpdateType(Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    UPDATE = "UPDATE"


    def __str__(self):
        return self.value

class CollateralUpdate:
    """
    Data class holding info about a collateral update
    """
    def __init__(self, amount: float, dt: datetime, message: str, collateral_update_type: CollateralUpdateType):
        """
        :param amount: The amount frozen in the account as collateral (Cannot be used to buy securities)
        :param dt: The datetime of the update.
        :param message: A message explaining the reason of the update.  (Useful for debugging the strategy)
        :param collateral_update_type: The type of update.  Can be ADD, REMOVE or UPDATE
        """
        self.amount = amount
        self.dt = dt
        self.message = message
        self.collateral_update_type = collateral_update_type

    def export(self) -> dict:
        """
        This method export the trade order object to a JSONable dictionary.
        :return: The object's state as a dictionary
        """
        return {
            "type": "CollateralUpdate",
            "amount": self.amount,
            "dt": str(self.dt),
            "message": self.message,
            "collateral_update_type": str(self.collateral_update_type)
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method load a CollateralUpdate object from a dictionary.
        :param data: The dictionary to load from
        :return: The CollateralUpdate object
        """
        return cls(data["amount"], datetime.fromisoformat(data["dt"]), data["message"],
                   CollateralUpdateType(data["collateral_update_type"]))

    def __eq__(self, other):
        return (self.amount == other.amount and self.dt == other.dt and self.message == other.message and
                self.collateral_update_type == other.collateral_update_type)

class Account:
    def __init__(self, initial_cash: float = 100_000, allow_debt: bool = False):
        self._cash = initial_cash
        self._collateral = 0    # This is the amount of money that is currently being used as collateral.
        self._allow_debt = allow_debt
        self._transactions = []
        self._collateral_history: List[CollateralUpdate] = []
        self._account_worth = [self._cash]
        self._previous_ids = set()
        self.n = 0    # Id for transactions

    def _update_worth(self):
        self._account_worth.append(self._cash)

    def deposit(self, amount: float, dt: datetime, transaction_id: str = None, comment: str = None):
        if transaction_id in self._previous_ids:
            raise RuntimeError("Transaction id already used!")
        else:
            if transaction_id is None:
                transaction_id = f'DefaultID: {self.n}'
            self._previous_ids.add(transaction_id)
        if amount < 0:
            raise RuntimeError("Deposit amount should be positive!")
        transaction = Transaction(amount, TransactionType.DEPOSIT, dt, transaction_id, comment)
        self._cash += transaction.amount
        if transaction.transaction_id is None:
            transaction.transaction_id = str(self.n)
        self._transactions.append(transaction)
        self.n += 1
        self._update_worth()

    def withdrawal(self, amount: float, dt: datetime, transaction_id: str = None, comment: str = None):
        transaction = Transaction(amount, TransactionType.WITHDRAWAL, dt, transaction_id, comment)
        if transaction.amount > self._cash and not self._allow_debt:
            raise RuntimeError("Transaction amount is bigger than current worth of account!")
        if transaction_id in self._previous_ids:
            raise RuntimeError("Transaction id already used!")
        else:
            if transaction_id is None:
                transaction_id = f'DefaultID: {self.n}'
            self._previous_ids.add(transaction_id)
        self._cash -= transaction.amount
        if transaction.transaction_id is None:
            transaction.transaction_id = str(self.n)
        self._transactions.append(transaction)
        self.n += 1
        self._update_worth()

    def update_collateral(self, amount: float, dt: datetime, message: str = "Step update"):
        """
        Updates the amount of collateral in the account.  This is the amount of money held as collateral and cannot
        be used.  This should be updated at each steps because it should be dependent to the current value of the
        assets.  Raise a RuntimeError if the amount of collateral is bigger than the current worth of the account.
        :param amount: Value of collateral.
        :param dt: datetime of the update
        :param message: Reason of the update.  Can be, for example: "Step update", "Enter short position for {ticker}", etc.
        :return: None
        """
        if amount > self._cash and not self._allow_debt:
            raise RuntimeError("Collateral amount is bigger than current worth of account!")
        self._collateral_history.append(CollateralUpdate(amount, dt, f"[UPDATE] - {message}",
                                                         CollateralUpdateType.UPDATE))
        self._collateral = amount

    def add_collateral(self, amount: float, dt: datetime, message: str = "Sold short"):
        """
        Adds collateral to the account.  This is the amount of money held as collateral and cannot
        be used.  This method could be used when selling short a position.
        :raise RuntimeError: If the amount of collateral is bigger than the current cash in the account.
        :param amount: Value of collateral.
        :param dt: datetime of the update
        :param message: Reason of the update.  Can be, for example: "Step update", "Enter short position for {ticker}", etc.
        :return: None
        """
        if amount + self._collateral > self._cash and not self._allow_debt:
            raise RuntimeError("Collateral amount is bigger than current worth of account!")
        self._collateral_history.append(CollateralUpdate(amount, dt, f"[ADD] - {message}",
                                                         CollateralUpdateType.ADD))
        self._collateral += amount

    def remove_collateral(self, amount: float, dt: datetime, message: str = "Bought back short"):
        """
        Removes collateral from the account.  This is the amount of money held as collateral and cannot
        be used.  This method could be used when buying back a short position.
        :raise RuntimeError: If the amount of collateral is bigger than the current cash in the account.
        :param amount: Value of collateral.
        :param dt: datetime of the update
        :param message: Reason of the update.  Can be, for example: "Step update", "Enter short position for {ticker}", etc.
        :return: None
        """
        if amount > self._collateral:
            raise RuntimeError("Collateral amount is bigger than current worth of account!")
        self._collateral_history.append(CollateralUpdate(amount, dt, f"[REMOVE] - {message}",
                                                         CollateralUpdateType.REMOVE))
        self._collateral -= amount

    def get_cash(self):
        """
        The amount of cash available in the account.  This is the amount of cash that can be used to buy securities.
        total_cash - collateral = cash
        :return: available cash
        """
        return self._cash - self._collateral


    def get_total_cash(self):
        """
        The total amount of cash in the account.  Not deducing collateral.  This include cash that cannot be used to
        buy securities.
        :return: Total cash
        """
        return self._cash

    def stats(self):
        return {
            "transactions": self._transactions,
            "cash_curve": self._account_worth,
            "final_cash": self._cash
        }

    def __str__(self):
        return f"BankAccount: {round(self.get_cash(), 2)}$"

    def get_state(self) -> dict:
        """
        This method export the account object to a JSONable dictionary.
        :return: The object state as a dictionary
        """
        return {
            "type": "Account",
            "cash": self._cash,
            "collateral": self._collateral,
            "transactions": [t.export() for t in self._transactions],
            "collateral_history": [c.export() for c in self._collateral_history],
            "account_worth": self._account_worth,
            "allow_debt": self._allow_debt,
            "previous_ids": list(self._previous_ids)
        }

    @classmethod
    def load_state(cls, dict_data: dict):
        """
        This method load a Account object from a dictionary.
        :param dict_data: The dictionary to load from
        :return: The Account object
        """
        account = cls()
        account._cash = dict_data["cash"]
        account._collateral = dict_data["collateral"]
        account._transactions = [Transaction.load(t) for t in dict_data["transactions"]]
        account._collateral_history = [CollateralUpdate.load(c) for c in dict_data["collateral_history"]]
        account._account_worth = dict_data["account_worth"]
        account._allow_debt = dict_data["allow_debt"]
        account._previous_ids = set(dict_data["previous_ids"])
        account.n = len(account._transactions)
        return account

    def __eq__(self, other):
        return (self._cash == other._cash and
                self._collateral == other._collateral and
                self._transactions == other._transactions and
                self._collateral_history == other._collateral_history and
                self._account_worth == other._account_worth and
                self._allow_debt == other._allow_debt and
                self._previous_ids == other._previous_ids and
                self.n == other.n)

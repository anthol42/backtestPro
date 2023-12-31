from .transaction import Transaction

class Account:
    def __init__(self, initial_cash: float = 100_000, allow_debt: bool = False):
        self._cash = initial_cash
        self._allow_debt = allow_debt
        self._transactions = []
        self._account_worth = []
        self.n = 0

    def _update_worth(self):
        self._account_worth.append(self._cash)

    def deposit(self, transaction: Transaction):
        self._cash += transaction.amount
        if transaction.transaction_id is None:
            transaction.transaction_id = str(self.n)
        self._transactions.append(transaction)
        self.n += 1
        self._update_worth()

    def withdrawal(self, transaction: Transaction):
        if transaction.amount > self._cash and not self._allow_debt:
            raise RuntimeError("Transaction amount is bigger than current worth of account!")
        self._cash -= transaction.amount
        if transaction.transaction_id is None:
            transaction.transaction_id = str(self.n)
        self._transactions.append(transaction)
        self.n += 1
        self._update_worth()

    def get_cash(self):
        return self._cash

    def stats(self):
        return {
            "transactions": self._transactions,
            "cash_curve": self._account_worth,
            "final_cash": self._cash
        }

    def __str__(self):
        return f"BankAccount: {round(self._cash, 2)}$"
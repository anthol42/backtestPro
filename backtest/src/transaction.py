from datetime import datetime
from enum import Enum

class TransactionType(Enum):
    DEPOSIT = 'DEPOSIT'
    WITHDRAWAL = 'WITHDRAWAL'

class Transaction:
    """
    Data class
    """
    def __init__(self, amount: float, T: TransactionType, dt: datetime, transaction_id: str = None, comment: str = None):
        self.amount = amount
        self.t = T
        self.dt = dt
        self.transaction_id = transaction_id
        self.comment = comment

    def __str__(self):
        if self.comment:
            return f"{self.t} {round(self.amount, 2)}$, {self.dt}, comment='{self.comment}'"
        else:
            return f"{self.t} {round(self.amount, 2)}$, {self.dt}"
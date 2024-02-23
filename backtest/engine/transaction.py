from datetime import datetime
from enum import Enum

class TransactionType(Enum):
    DEPOSIT = 'DEPOSIT'
    WITHDRAWAL = 'WITHDRAWAL'

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str):
        if string == 'DEPOSIT':
            return cls.DEPOSIT
        elif string == 'WITHDRAWAL':
            return cls.WITHDRAWAL
        else:
            raise ValueError(f"Invalid string: {string}")

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

    def export(self) -> dict:
        """
        This method export the trade order object to a JSONable dictionary.
        :return: The object's state as a dictionary
        """
        return {
            "type": f"Transaction.{self.t.value}",
            "amount": self.amount,
            "dt": str(self.dt),
            "transaction_id": self.transaction_id,
            "comment": self.comment,
            "t": self.t.value
        }
    @classmethod
    def load(cls, data: dict):
        """
        This method load a Transaction object from a dictionary.
        :param data: The dictionary to load from
        :return: The Transaction object
        """
        return cls(data["amount"], TransactionType.from_str(data["t"]), datetime.fromisoformat(data["dt"]),
                   data["transaction_id"], data["comment"])


    def __eq__(self, other):
        return self.amount == other.amount and self.t == other.t and self.dt == other.dt and self.transaction_id == other.transaction_id and self.comment == other.comment
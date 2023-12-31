import pandas as pd


class BackTest:
    def __init__(self, data: pd.DataFrame, initial_cash: float = 100_000, buy_on_close: bool = False, commission: float = None,
                 relative_commission: float = None):
        self._data = data
        self._initial_cash = initial_cash
        self._bonc = buy_on_close
        self._comm = commission
        self._rel_comm = relative_commission


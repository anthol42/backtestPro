from ..engine import Account, Broker, Portfolio, TradeOrder, TradeType, TSData, Strategy, CashControllerBase
from typing import Any, Optional, List, Dict, Union, Tuple
from datetime import datetime
from enum import Enum

class ServerStatus(Enum):
    OK = "Ok"
    WARNING = "Warning"
    ERROR = "Error"

    def __str__(self):
        return self.value

class StateSignals:
    """
    Class that stores the signals given y the strategy for the current timestep and state of the account, the broker
    and the portfolio.
    """

    def __init__(self, account: Account, broker: Broker, signals: Dict[str, TradeOrder], strategy: Strategy,
                 timestamp: datetime, cash_controller: CashControllerBase, initial_cash: float,
                 index_data: Optional[Dict[str, TSData]] = None, data: Optional[List[Dict[str, TSData]]] = None,
                 main_idx: Optional[int] = None, backtest_params: Optional[Dict[str, Any]] = None,
                 status: ServerStatus = ServerStatus.OK):
        self.account = account
        self.broker = broker
        self.portfolio = broker.portfolio    # An alias for easier access
        self.strategy = strategy
        self.cash_controller = cash_controller
        self.initial_cash = initial_cash
        self._signals = signals
        self.timestamp = timestamp
        self.index_data = index_data
        self.data = data
        self.main_idx = main_idx
        self.backtest_params = backtest_params    # The parameters passed to the backtest object.
        self.status = status


    @property
    def buy_long_signals(self) -> Dict[str, TradeOrder]:
        """
        Returns the buy long signals
        """
        return {k: v for k, v in self._signals.items() if v.trade_type == TradeType.BuyLong}

    @property
    def sell_long_signals(self) -> Dict[str, TradeOrder]:
        """
        Returns the sell long signals
        """
        return {k: v for k, v in self._signals.items() if v.trade_type == TradeType.SellLong}

    @property
    def buy_short_signals(self) -> Dict[str, TradeOrder]:
        """
        Returns the buy short signals
        """
        return {k: v for k, v in self._signals.items() if v.trade_type == TradeType.BuyShort}

    @property
    def sell_short_signals(self) -> Dict[str, TradeOrder]:
        """
        Returns the sell short signals
        """
        return {k: v for k, v in self._signals.items() if v.trade_type == TradeType.SellShort}

    def __getitem__(self, item: str) -> TradeOrder:
        if item not in self._signals:
            raise KeyError(f"No signals were emitted for ticker {item}")
        return self._signals[item]

    def get(self, item: str, default: Optional[Any] = None) -> Optional[TradeOrder]:
        return self._signals.get(item, default)

    def __iter__(self):
        return iter(self._signals.values())


    def __repr__(self):
        return f"StateSignals(count={len(self._signals)})"

    def __str__(self):
        s = ""
        l = []
        max_len = 0
        if len(self._signals) == 0:
            s = "No signals were emitted"
        else:
            for ticker, order in self._signals.items():
                if len(ticker) + len(order.trade_type.value) > max_len:
                    max_len = len(ticker) + len(order.trade_type.value)
                l.append((ticker, order.trade_type.value))

            if len(self._signals) < 10:
                for ticker, order in l:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}: {' ' * (max_len - curr_len + 1)}{order}\n"
            else:
                for ticker, order in l[:5]:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}: {' ' * (max_len - curr_len + 1)}{order}\n"
                s += "    ...\n"
                for ticker, order in l[-5:]:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}: {' ' * (max_len - curr_len + 1)}{order}\n"

        # Put the output in a bounding box
        lines = s.split("\n")
        if lines[-1] == "":
            lines = lines[:-1]
        width = max(len(line) for line in lines)
        top_line = f"┌ {len(self._signals)} Signals "
        top_line = top_line.ljust(width + 3, "─") + "┐"
        width = len(top_line) - 4
        bottom_line = f"└{'─' * (len(top_line) - 2)}┘"
        lines = ["│ " + line.ljust(width) + " │" for line in [""] + lines + [""]]
        return "\n".join([top_line] + lines + [bottom_line])



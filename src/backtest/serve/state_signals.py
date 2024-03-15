from ..engine import Account, Broker, Portfolio, TradeOrder, TradeType, TSData
from typing import Any, Optional, List, Dict, Union, Tuple
from datetime import datetime

class StateSignals:
    """
    Class that stores the signals given y the strategy for the current timestep and state of the account, the broker
    and the portfolio.
    """

    def __init__(self, account: Account, broker: Broker, signals: Dict[str, TradeOrder], timestamp: datetime,
                 index_data: Optional[List[Dict[str, TSData]]] = None):
        self.account = account
        self.broker = broker
        self.portfolio = broker.portfolio    # An alias for easier access
        self._signals = signals
        self.timestamp = timestamp
        self.index_data = index_data


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
        return iter(self._signals)


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
                if len(ticker) + len(str(order.trade_type)) > max_len:
                    max_len = len(ticker) + len(str(order.trade_type))
                l.append((ticker, str(order.trade_type)))

            if len(self._signals) < 10:
                for ticker, order in l:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}{' ' * (max_len - curr_len + 1)}: {order}\n"
            else:
                for ticker, order in l[:5]:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}{' ' * (max_len - curr_len + 1)}: {order}\n"
                s += "    ...\n"
                for ticker, order in l[-5:]:
                    curr_len = len(ticker) + len(order)
                    s += f"{ticker}{' ' * (max_len - curr_len + 1)}: {order}\n"

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



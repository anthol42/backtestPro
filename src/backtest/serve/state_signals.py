# Copyright (C) 2024 Anthony Lavertu
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from ..engine import Account, Broker, Portfolio, TradeOrder, TradeType, TSData, Strategy, CashControllerBase
from typing import Any, Optional, List, Dict, Union, Tuple
from datetime import datetime
from enum import Enum
from warnings import WarningMessage

class ServerStatus(Enum):
    """
    Used to indicate the status of the server after the strategy has been executed.
    """
    OK = "Ok"
    WARNING = "Warning"
    ERROR = "Error"

    def __str__(self):
        return self.value

class StateSignals:
    """
    Class that stores the signals given y the strategy for the current timestep and state of the account, the broker
    and the portfolio.  Its more of a dataclass that stores the signals and the state of the account,
    broker and portfolio.  It also has few method for easy access to the signals.
    """

    def __init__(self, account: Account, broker: Broker, signals: Dict[str, TradeOrder], strategy: Strategy,
                 timestamp: datetime, cash_controller: CashControllerBase, initial_cash: float,
                 index_data: Optional[Dict[str, TSData]] = None, data: Optional[List[Dict[str, TSData]]] = None,
                 main_idx: Optional[int] = None, backtest_params: Optional[Dict[str, Any]] = None,
                 status: ServerStatus = ServerStatus.OK, exception: Optional[Exception] = None,
                 warnings: Optional[List[WarningMessage]] = None):
        """
        :param account: The account object
        :param broker: The broker object
        :param signals: The signals emitted by the strategy
        :param strategy: The strategy object
        :param timestamp: The current timestamp (i.e. the time at which the strategy was executed)
        :param cash_controller: The cash controller object
        :param initial_cash: The initial cash in the account (Useful to calculate the returns in a renderer)
        :param index_data: The index data for the current timestamp.  Same time-resolution as the main time resolution.
        :param data: The data for the current timestamp.
        :param main_idx: The index of the main time resolution in the data list.
        :param backtest_params: The backtest parameters passed as a dictionary.
        :param status: The server status after the strategy has been executed.
        :param exception: An exception object if an exception was raised during the strategy execution.
        :param warnings: Any warning objects that were raised during the strategy execution.
        """
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
        self.warnings = warnings
        self.exception = exception


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
        """
        Returns the signal for the given ticker
        :param item: The ticker to search for
        :return: The TradeOrder object
        :raises KeyError: If no signals were emitted for the given ticker
        """
        if item not in self._signals:
            raise KeyError(f"No signals were emitted for ticker {item}")
        return self._signals[item]

    def get(self, item: str, default: Optional[Any] = None) -> Union[TradeOrder, Any]:
        """
        Returns the signal for the given ticker
        :param item: The ticker to search for
        :param default: The default value to return if no signals were emitted for the given ticker
        :return: The TradeOrder object or the default value
        """
        return self._signals.get(item, default)

    def __iter__(self):
        """
        Iterates over the signals
        """
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



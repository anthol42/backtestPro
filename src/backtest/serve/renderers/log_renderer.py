"""
Copyright (C) 2024 Anthony Lavertu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
from pathlib import PurePath
from ..renderer import Renderer
from ..state_signals import StateSignals
from ...engine import TradeOrder
from typing import Union, List
import os

class LogRenderer(Renderer):
    """
    Append signals to a log file. (csv format)
    The columns are:
    - timestamp
    - security
    - signal_type
    - price_lower_limit
    - price_upper_limit
    - n_shares
    - n_shares_borrowed
    - expiry
    """
    def __init__(self, sub_dir: Union[PurePath, str] = "signals"):
        super().__init__()
        self.sub_dir = sub_dir if isinstance(sub_dir, PurePath) else PurePath(sub_dir)

    def render(self, state: StateSignals, base_path: PurePath):
        signal_list: List[TradeOrder] = list(state.buy_long_signals.values()) + list(state.sell_long_signals.values()) +\
                       list(state.buy_short_signals.values()) + list(state.sell_short_signals.values())

        if not os.path.exists(base_path / self.sub_dir):
            os.makedirs(base_path / self.sub_dir)

        if not os.path.exists(base_path / self.sub_dir / "signals.log"):
            with open(base_path / self.sub_dir / "signals.log", "w") as f:
                f.write("timestamp,security,signal_type,price_lower_limit,price_upper_limit,n_shares,n_shares_borrowed,expiry\n")

        with open(base_path / self.sub_dir / "signals.log", "a") as f:
            for signal in signal_list:
                f.write(f"{state.timestamp.isoformat()},{signal.security},{signal.trade_type.value},"
                        f"{signal.security_price_limit[0]},{signal.security_price_limit[1]},{signal.amount},"
                        f"{signal.amount_borrowed},{signal.expiry}\n")

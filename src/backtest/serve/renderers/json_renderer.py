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
import json
from pathlib import PurePath
from ..renderer import Renderer
from ..state_signals import StateSignals
from typing import Union
import os

class JSONRenderer(Renderer):
    """
    This renderer render the signals in JSON format.  You can also store the portfolio state, the broker state and the
    account state in the json.
    """

    def __init__(self, sub_dir: Union[PurePath, str] = "signals", store_portfolio: bool = False, store_broker: bool = False,
                 store_account: bool = False):
        """
        :param sub_dir: The sub directory where the json file will be stored.
        :param store_portfolio: Whether to store the portfolio state in the json.
        :param store_broker: Whether to store the broker state in the json.
        :param store_account: Whether to store the account state in the json.
        """
        super().__init__()
        self.store_portfolio = store_portfolio
        self.store_broker = store_broker
        self.store_account = store_account
        self.sub_dir = sub_dir if isinstance(sub_dir, PurePath) else PurePath(sub_dir)

    def render(self, state: StateSignals, base_path: PurePath):
        out = {
            "timestamp": state.timestamp.isoformat(),
            "signals": {
                "buy_long": {k: v.export() for k, v in state.buy_long_signals.items()},
                "sell_long": {k: v.export() for k, v in state.sell_long_signals.items()},
                "buy_short": {k: v.export() for k, v in state.buy_short_signals.items()},
                "sell_short": {k: v.export() for k, v in state.sell_short_signals.items()},
            }
        }
        if self.store_portfolio:
            out["portfolio"] = state.portfolio.get_state()

        if self.store_broker:
            out["broker"] = state.broker.get_state()

        if self.store_account:
            out["account"] = state.account.get_state()

        if not os.path.exists(base_path / self.sub_dir):
            os.makedirs(base_path / self.sub_dir)
        with open(base_path / self.sub_dir / "signals.json", "w") as f:
            json.dump(out, f)


"""
This module contains the logic to run the backtest.
"""
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
from .backtest import Backtest
from .time_resolution_extenders import TimeResExtender, BasicExtender
from .broker import Broker, MarginCall, BrokerState, StepState
from .portfolio import Portfolio, Position, TradeStats
from .account import Account, CollateralUpdateType, CollateralUpdate
from .transaction import Transaction, TransactionType
from .trade import Trade, TradeOrder, TradeType
from .trade import BuyLongOrder, SellLongOrder, BuyShortOrder, SellShortOrder
from .trade import BuyLong, SellLong, BuyShort, SellShort
from .backtestResult import BackTestResult
from .cashController import CashControllerBase, SimpleCashController, CashControllerTimeframe
from .strategy import Strategy
from .metadata import Metadata
from .record import Record, Records, RecordsBucket
from .tsData import TSData, DividendFrequency
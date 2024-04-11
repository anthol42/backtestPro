"""
This module contains multiple helper function to execute a backtest.
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
import pandas as pd

def crossover(series1: pd.Series, series2: pd.Series):
    return (series1.iloc[-1] > series2.iloc[-1]) & (series1.iloc[-2] < series2.iloc[-2])

def descending(series: pd.Series):
    return series.iloc[-1] < series.iloc[-2]
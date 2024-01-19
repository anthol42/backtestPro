import pandas as pd
from typing import List, Dict, Union, Callable, Tuple
from src.account import Account
from src.broker import Broker
import numpy as np
from datetime import datetime, timedelta
from src.tsData import TSData
from src.strategy import Strategy
from typing import List, Dict, Type
from src.backtestResult import BackTestResult
from tqdm import tqdm


class BackTest:
    def __init__(self, data: List[Dict[str, TSData]], strategy: Type[Strategy], *, main_timestep: int = 0,
                 initial_cash: float = 100_000,
                 commission: float = None,
                 relative_commission: float = None, margin_interest: float = 0,
                 min_initial_margin: float = 0.5, min_maintenance_margin: float = 0.25,
                 liquidation_delay: int = 2, min_initial_margin_short: float = 0.5,
                 min_maintenance_margin_short: float = 0.25,
                 broker: Type[Broker] = Broker, account: Type[Account] = Account):

        self._data = data
        self._initial_cash = initial_cash
        self.account = account(initial_cash)
        self.broker = broker(self.account, commission, relative_commission, margin_interest, min_initial_margin,
                             min_maintenance_margin, liquidation_delay, min_initial_margin_short,
                             min_maintenance_margin_short)
        self.strategy = strategy(self.account, self.broker)
        self.main_timestep = main_timestep    # The index of the timeseries data in data list to use as the main series.
                                              # i.e. the frequency at hich the strategy is runned.
                                              # The other timeseries will be passed to the strategy as additional data.

    def _step(self, i: int, timestep: datetime):
        # Step 1: Select stocks that are available at the current timestep
        # Step 2: Prepare the data - including cropping candles that have lower frequency than the main timestep frequency
        # Step 3: Run strategy
        # Step 4: Run broker
        pass

    def run(self) -> BackTestResult:
        """
        Run the backtest simulation with the given data and strategy.
        It will show a progress bar showing the current progress.
        It will save the results and all the statistics into a BackTestResult object in the results attribute, or
        returned value of this method.
        :return: BackTestResult object containing all the results, statistics, debug info, etc.
        """

        # Step 1: Initialization
        # Initialize strategy
        self.strategy.init()
        timesteps_list = []
        # Evaluate if data is valid + make timestep list.
        available_time_res = []
        for i, ts_group in enumerate(self._data):
            available_time_res.append(ts_group[ts_group.keys()[0]].time_res)
            # Check if all the data has the same time resolution
            for ticker, ts in ts_group.items():
                if ts.time_res != available_time_res[i]:
                    raise ValueError(f"All the timeseries data must have the same time resolution.\n"
                                     f"Ticker: {ticker} for data in group {i} has a different time resolution than the "
                                     f"other data in the same group.")
                if i == self.main_timestep:
                    if len(ts.data) > len(timesteps_list):
                        timesteps_list = list(ts.data.index)

        # Step 2: Run simulation
        for i, timestep in enumerate(tqdm(timesteps_list, desc="Backtesting...")):
            self._step(i, timestep)


        # Step 3: Prepare and save stats




import pandas as pd
from typing import List, Dict, Union, Callable, Tuple
from src.account import Account
from src.broker import Broker
import numpy as np
from datetime import datetime, timedelta
from src.tsData import TSData
from src.strategy import Strategy
from typing import List, Dict, Type, Optional
from src.backtestResult import BackTestResult
from src.record import Record
from tqdm import tqdm
import warnings

class UnexpectedBehaviorRisk(Warning):
    pass

class BackTest:
    def __init__(self, data: List[Dict[str, TSData]], strategy: Type[Strategy], *, main_timestep: int = 0,
                 initial_cash: float = 100_000,
                 commission: float = None,
                 relative_commission: float = None, margin_interest: float = 0,
                 min_initial_margin: float = 0.5, min_maintenance_margin: float = 0.25,
                 liquidation_delay: int = 2, min_initial_margin_short: float = 0.5,
                 min_maintenance_margin_short: float = 0.25,
                 broker: Type[Broker] = Broker, account: Type[Account] = Account,
                 window: int = 50, default_marginable: bool = False,
                 default_shortable: bool = False):

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
        if window < 1:
            raise ValueError("Window must be at least 1.")
        self.window = window
        # Use this value if it is not specified in the data (Column Marginable)
        self.default_marginable = default_marginable
        # Use this value if it is not specified in the data (Column Shorable)
        self.default_shortable = default_shortable

    def _step(self, i: int, timestep: datetime):
        # Step 1: Prepare the data
        time_res = self.available_time_res[self.main_timestep]
        prepared_data: List[Optional[List[Record]]] = [None for _ in range(len(self.available_time_res))]

        # Prepare the data for main timestep
        prepared_data[self.main_timestep] = self._prepare_data(self._data, self.main_timestep, timestep)

        # Check if there exists time resolution bigger than the main timesteps.  In that case, we need to crop the data.
        time_res_bigger_idx = [i for i, res in enumerate(self.available_time_res) if res > time_res]
        time_res_smaller_idx = [i for i, res in enumerate(self.available_time_res) if res < time_res]
        for idx in time_res_smaller_idx:
            prepared_data[idx] = self._prepare_data(self._data, idx, timestep)

        # Forge new candles for the one with bigger time resolution  to avoid peeking into the future.  (Data leaking)
        for idx in time_res_bigger_idx:
            series = self._prepare_data(self._data, idx, timestep)
            last_candles = self.forge_last_candle(self._data, prepared_data, idx, timestep)
            for i, record in enumerate(series):
                if record.chart is not None:
                    record.chart.iloc[-1] = pd.Series(last_candles[i][:-1], index=["Open", "High", "Low", "Close", "Volume"])
                    assert record.ticker == last_candles[i][:-1], "An error happened, securities are no longer aligned"
            prepared_data[idx] = series

        # Step 2: Filter stock data to remove the ones that are not currently available (Chart is None for main res)
        mask = np.ones(len(prepared_data[self.main_timestep]), dtype=bool)
        for idx, record in enumerate(prepared_data[self.main_timestep]):
            if record.chart is None:
                mask[idx] = False    # Flip the mask to remove the record
        prepared_data = [np.array(prepared_data[i])[mask].tolist() for i in range(len(prepared_data))]

        # Step 3: Run strategy
        self.strategy(prepared_data, timestep)
        # Step 4: Run broker
        # Get security names
        security_names = [record.ticker for record in prepared_data[self.main_timestep]]
        # Prepare current data for broker
        current_data = np.array([record.chart.iloc[-1].to_list() for record in prepared_data[self.main_timestep]], dtype=np.float32)
        next_tick_data = np.array([record.next_tick.to_list() for record in prepared_data[self.main_timestep]], dtype=np.float32)
        marginables = np.array([[record.marginable, record.shortable] for record in prepared_data[self.main_timestep]], dtype=np.bool)
        self.broker.tick(timestep, security_names, current_data, next_tick_data, marginables)

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
        available_time_res: List[timedelta] = []
        # Many groups of time series
        for i, ts_group in enumerate(self._data):
            available_time_res.append(ts_group[ts_group.keys()[0]].time_res)
            # Check if all the data has the same time resolution
            last_index = None
            for ticker, ts in ts_group.items():
                if ts.time_res != available_time_res[i]:
                    raise ValueError(f"All the timeseries data must have the same time resolution.\n"
                                     f"Ticker: {ticker} for data in group {i} has a different time resolution than the "
                                     f"other data in the same group.")
                if last_index is None:
                    last_index = ts.data.index
                else:
                    if last_index!= ts.data.index:
                        raise ValueError(f"All the timeseries data in the same group must have the same index.\n"
                                            f"Ticker: {ticker} for data in group {i} has a different index than the "
                                            f"other data in the same group.")
                if i == self.main_timestep:
                    if len(ts.data) > len(timesteps_list):
                        timesteps_list = list(ts.data.index)
        self.available_time_res = available_time_res

        # Optionnaly reformat timesteps_list.  Can be useful when indexes are variable across stocks
        timesteps_list = self.stadardize_timesteps(timesteps_list)
        # Step 2: Run simulation
        for i, timestep in enumerate(tqdm(timesteps_list, desc="Backtesting...")):
            self._step(i, timestep)


        # Step 3: Prepare and save stats




    def _prepare_data(self, data: List[Dict[str, TSData]], current_time_res: int, timestep: datetime) -> List[Record]:
        """
        Prepare the data for the current timestep.  This method assumes that nan are padding.  This means that it is
        assumed that the security didn't exist at the time where there is nan.  It is important to fill nan in data
        preprocessing steps before starting the backtest.  If you do not want to impute value, you can override this
        method.
        :param data: The data
        :param current_time_res: The current time resolution
        :param timestep: The current timestep
        :return: The prepared data
        """
        prepared_data: List[Record] = []
        for ticker, ts in data[current_time_res].items():
            cropped = ts.data.loc[:timestep]
            timestep_idx = ts.data.index.get_loc(timestep)

            # Find start padding
            start_idx = np.argmin(np.isnan(cropped[["Open", "High", "Low", "Close", "Volume"]].to_numpy()).any(axis=1))

            # Find end padding
            end_idx = start_idx + np.argmax(
                np.isnan(cropped[["Open", "High", "Low", "Close", "Volume"]].to_numpy()[start_idx:]).any(axis=1))

            # The security does not exist yet
            if end_idx == start_idx:
                prepared_data.append(Record(None, ticker, current_time_res))
                continue

            # Security has been delisted or there is missing values.  (We will ignore it)
            if end_idx != timestep_idx:
                prepared_data.append(Record(None, ticker, current_time_res))
                continue

            # Check if window is too big
            if end_idx - start_idx > self.window:
                start_idx = end_idx - self.window

            if "Marginable" in ts.data.columns:
                marginable = ts.data["Marginable"].iloc[end_idx - 1]
            else:
                marginable = self.default_marginable
            if "Shortable" in ts.data.columns:
                shortable = ts.data["Shortable"].iloc[end_idx - 1]
            else:
                shortable = self.default_shortable
            prepared_data.append(Record(cropped.iloc[start_idx:], ticker, current_time_res,
                                        marginable, shortable,
                                        ts.data[["Open", "High", "Low", "Close", "Volume"]].iloc[end_idx]))

        return prepared_data

    def stadardize_timesteps(self, timesteps_list: List[datetime]) -> List[datetime]:
        """
        Optionnaly reformat timesteps_list.  Can be useful when indexes are variable across stocks
        :param timesteps_list: The list of timesteps
        :return: The reformatted list of timesteps
        """
        return timesteps_list


    def forge_last_candle(self, data: List[Dict[str, TSData]], prepared_data: List[List[Record]], current_time_res: int,
                      timestep: datetime) -> List[Tuple[float, float, float, float, float, str]]:
        """
        Forge new candle by cropping the last candle for the data with bigger time resolution than the main timestep to avoid
        peeking into the future.  (Data leaking)
        How this default method works: (Read the warning below if your data contains higher resolution than main res)

        For each Tickers:
        1. Find the last candle compared to current time step with the current resolution. (Get index and Open)
        2. Find the last candle for the ticker with main resolution.  (Get index and Close)
        3. Using the main resolution, get the series between the beginning of the candle (Current resolution) and the
            current timestep.  (This will be a series of intra-candle data)  This is where it can get tricky because,
            depending on your data, the start idx (computed in step 1) might not align with any timesteps of the series
            in the main resolution.  In this default method, we will use the next candle the nearest of the desired
            start idx of the main resolution data.  (Default behavior of pandas)
        4. Get High by computing the max of the maxes of the intra-candle data and the Low by computing the min of the
            mins of the intra-candle data.  The volume is obtained by summing the volume
            of the intra-candle data.
        5. Return the new candle by using computed High Low Open Close and Volume.

        IMPORTANT:
            If you need this method in your setup, you should override it since it may not work for your setup.
            In fact, it is hard to forge new candles for any arbitrary time resolution from discrete timeseries where
            timesteps might not correctly align.  You should take this into account when making this method.

        TO OVERRIDE THIS METHOD:
            Just override it like any other method.  You can also use the default method by calling
            'self.default_forge_last_candle' and complement it.  To suppress the warning, you can override this method
            by only calling the default 'self.default_forge_last_candle'

        :param data: The data
        :param prepared_data: The already prepared data.
        :param current_time_res: The current time resolution
        :param timestep: The current timestep
        :return: A list of the forged candles: List[Tuple[Open, High, Low, Close, Volume, ticker]]
        """
        warnings.warn("This method is not guaranteed to work for your setup.  You should override it, or make sure it "
                      "works for your setup if you have series that have a higher resolution than the main resolution.",
                      UnexpectedBehaviorRisk)
        return self.default_forge_last_candle(data, prepared_data, current_time_res, timestep)


    def default_forge_last_candle(self, data: List[Dict[str, TSData]], prepared_data: List[List[Record]], current_time_res: int,
                      timestep: datetime) -> List[Tuple[float, float, float, float, float, str]]:
        """
        Forge new candle by cropping the last candle for the data with bigger time resolution than the main timestep to avoid
        peeking into the future.  (Data leaking)
        How this default method works:

        For each Tickers:
        1. Find the last candle compared to current time step with the current resolution. (Get index and Open)
        2. Find the last candle for the ticker with main resolution.  (Get index and Close)
        3. Using the main resolution, get the series between the beginning of the candle (Current resolution) and the
            current timestep.  (This will be a series of intra-candle data)  This is where it can get tricky because,
            depending on your data, the start idx (computed in step 1) might not align with any timesteps of the series
            in the main resolution.  In this default method, we will use the next candle the nearest of the desired
            start idx of the main resolution data.  (Default behavior of pandas)
        4. Get High by computing the max of the maxes of the intra-candle data and the Low by computing the min of the
            mins of the intra-candle data.  The volume is obtained by summing the volume
            of the intra-candle data.
        5. Return the new candle by using computed High Low Open Close and Volume.

        DO NOT OVERRIDE THIS METHOD:
            If you need to override a candle forging method, override the method 'forge_last_candle' instead.
        :param data: The data
        :param prepared_data: The already prepared data.
        :param current_time_res: The current time resolution
        :param timestep: The current timestep
        :return: A list of the forged candles: List[Tuple[Open, High, Low, Close, Volume, ticker]]
        """
        newly_forged_candles = []
        for ticker, ts in data[current_time_res].items():
            # Step 1: Find the last candle compared to current time step with the current resolution. (Get index and Open)
            last_candle = ts.data.loc[:timestep].iloc[-1]
            candle_open = last_candle["Open"]
            start_index = last_candle.name

            # Step 2: Find the last candle for the ticker with main resolution.  (Get index and Close)
            main_res_data = data[self.main_timestep][ticker].data
            main_res_last_candle = main_res_data.data.loc[:timestep].iloc[-1]
            candle_close = main_res_last_candle["Close"]
            end_index = main_res_last_candle.name

            # Step 3: Get the series between the beginning of the candle (Current resolution) and the current timestep.
            intra_candle = main_res_data.loc[start_index:end_index]

            # Step 4: Get missing OHLC stats: High, Low and Volume
            candle_high = intra_candle["High"].max()
            candle_low = intra_candle["Low"].min()
            candle_volume = intra_candle["Volume"].sum()

            # Step 5: Add this to new forged candles
            newly_forged_candles.append((candle_open, candle_high, candle_low, candle_close, candle_volume, ticker))

        return newly_forged_candles
import pandas as pd
from typing import Tuple
from .account import Account
from .broker import Broker
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta
from .tsData import TSData
from .strategy import Strategy
from typing import List, Dict, Type, Optional, Union
from .backtestResult import BackTestResult
from .record import Record, RecordsBucket
from .tsData import DividendFrequency
from tqdm import tqdm
import warnings
from .metadata import Metadata
from .cashController import CashControllerBase, CashControllerTimeframe
from .time_resolution_extenders import TimeResExtender
from ..indicators import IndicatorSet

class UnexpectedBehaviorRisk(Warning):
    pass

class Backtest:
    def __init__(self, data: List[Dict[str, TSData]], strategy: Strategy, *, metadata: Metadata = Metadata(),
                 market_index: TSData = None,
                 main_timestep: int = 0,
                 initial_cash: float = 100_000,
                 commission: float = None,
                 relative_commission: float = None, margin_interest: float = 0,
                 min_initial_margin: float = 50, min_maintenance_margin: float = 25,
                 liquidation_delay: int = 2, min_initial_margin_short: float = 50,
                 min_maintenance_margin_short: float = 25,
                 broker: Type[Broker] = Broker, account: Type[Account] = Account,
                 window: int = 50, default_marginable: bool = False,
                 default_shortable: bool = False,
                 risk_free_rate: float = 1.5,
                 default_short_rate: float = 1.5,
                 sell_at_the_end: bool = True,
                 cash_controller: CashControllerBase = CashControllerBase(),
                 verbose=3,
                 time_res_extender: Optional[TimeResExtender] = None,
                 indicators: Union[IndicatorSet, List[IndicatorSet], Dict[int, IndicatorSet]] = IndicatorSet(),
                 streaming_indicators: bool = False):
        """
        :param data: The data on which to run the backtest.  It is a list of dictionaries where each dictionary
                        represents a group of time series data.  The key is the ticker and the value is the TSData object.
        :param strategy: The strategy to test
        :param metadata: A metadata object.  This is useful to store information about the backtest to add context.
        :param market_index: The TSData object containing the reference market index.  (Used to compare the strategy
                            performances to the market)
        :param main_timestep: The index of the timeseries data in data list to use as the main series.  i.e. the
                            frequency at which the strategy is run.  The other timeseries will be passed to the
                            strategy as additional data.  For example, if our data is a list of two dictionaries,
                            the first one containing hourly data and the second one containing daily data, and we
                            extend the daily data to weekly data, we would set main_timestep to 1.  (The second
                            dictionary) to run our strategy at each open market days.
        :param initial_cash: The initial cash in bank account
        :param commission: The absolute commission.  In $ per trade
        :param relative_commission: The relative commission.  In % of the trade value.  Example: 5% would be 5
        :param margin_interest: The interest rate on margin.  In % of the margin value.  Example: 5% would be 5
        :param min_initial_margin: The minimum initial margin.  In % of the trade value.  Example: 5% would be 5
        :param min_maintenance_margin: The minimum maintenance margin.  In % of the trade value.  Example: 5% would be 5
        :param liquidation_delay: The delay before liquidating a position in margin call.  In timestep.  Example, if the
                                delay is 2 and the main time resolution is daily, the position will be liquidated 2 days
        :param min_initial_margin_short: The minimum initial margin for short positions.  In % of the trade value.
        :param min_maintenance_margin_short: The minimum maintenance margin for short positions.  In % of the trade value.
        :param broker: The Broker class to use.  This is useful to use a custom broker.  The broker class must inherit
                    from the Broker class.
        :param account: The Account class to use.  This is useful to use a custom account.  The account class must inherit
                    from the Account class.
        :param window: The lookback period for the strategy.  The number of datapoint to pass to the strategy at each
                    timestep.  (The number of datapoint in the main timestep.  There might be more or less for others)
        :param default_marginable: In case there is no column "Marginable" in the data, this value will be used to know
                    if the security is marginable.
        :param default_shortable: In case there is no column "Shortable" in the data, this value will be used to know if
                    the security is shortable.
        :param risk_free_rate: The risk-free rate in %.  Used to calculate performance ratios like the Sharpe ratio.
        :param default_short_rate: The short rate in % in case there is no column 'Short_rate' in the data.
        :param sell_at_the_end: Whether to sell every open position at the end or not.  If True, the strategy will
                    sell the positions at the end of the backtest.
        :param cash_controller: A cash controller object to control the money flow in and out of the bank account during
                    the backtest.  This is useful to simulate the effect of, for example, weekly deposit or monthly
                    withdrawal.
        :param verbose: 0: No print, 1: Only errors, 2: Errors and warnings, 3: All
        :param time_res_extender: A time resolution extender object.  This is useful to add time resolutions to the data
        :param indicators: An Indicator set object.  This is useful to add indicators to the data.  The indicators are
                    calculated at each timestep and passed to the strategy.   It can also be a list with the same length
                    as the number of time resolution.  Each element of the list will be an IndicatorSet object to use for
                    the corresponding time resolution.  If it is a dictionary, the keys will be the index of the time
                    resolution and the values will be the IndicatorSet object to use for the corresponding time resolution.
                    If a dictionary is used, make sure there is at least a key for the main time resolution.
        """


        self._data = data
        self._initial_cash = initial_cash
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        self.account = account(initial_cash)
        self.broker = broker(self.account, commission, relative_commission and relative_commission / 100, margin_interest / 100,
                             min_initial_margin / 100, min_maintenance_margin / 100, liquidation_delay,
                             min_initial_margin_short / 100, min_maintenance_margin_short / 100)
        self.strategy = strategy
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
        self.default_short_rate = default_short_rate / 100
        self.metadata = metadata
        self.sell_at_the_end = sell_at_the_end
        self.indicators = indicators
        self.streaming_indicators = self.indicators.streaming
        self._backtest_parameters = {
            "strategy": strategy.__class__.__name__,
            "main_timestep": main_timestep,
            "initial_cash": initial_cash,
            "commission": commission,
            "relative_commission": relative_commission,
            "margin_interest": margin_interest,
            "min_initial_margin": min_initial_margin,
            "min_maintenance_margin": min_maintenance_margin,
            "liquidation_delay": liquidation_delay,
            "min_initial_margin_short": min_initial_margin_short,
            "min_maintenance_margin_short": min_maintenance_margin_short,
            "window": window,
            "default_marginable": default_marginable,
            "default_shortable": default_shortable,
            "risk_free_rate": risk_free_rate,
            "default_short_rate": default_short_rate,
            "sell_at_the_end": sell_at_the_end,
            "cash_controller": cash_controller.__class__.__name__,
            "indicators": indicators.toList(),
            "streaming_indicators": streaming_indicators,
            "verbose": verbose,
            "time_res_extender": time_res_extender.export() if time_res_extender is not None else None
        }
        # Resolutions are added at the end of the Record objet list.
        self.time_res_extender = time_res_extender
        self.cash_controller = cash_controller
        self._verbose = verbose    # 0: No print, 1: Only errors, 2: Errors and warnings, 3: All
        self.run_iter: int = 0
        # This is used to cache the prepared data after each step.  This is useful to avoid recomputing the same data
        # as when using the streaming mode of indicators.
        self.cache_data: List[Dict[str, pd.DataFrame]] = []

    def step(self, i: int, timestep: datetime, next_time_step: datetime):
        # Step 1: Prepare the data
        processed_data: List[List[Record]] = self._prep_data(timestep)

        # Step 2: Filter stock data to remove the ones that are not currently available (Chart is None for main res)
        mask = self._get_mask(processed_data[self.main_timestep])
        filtered_data: List[npt.NDArray[Record]] = [np.array(processed_data[i])[mask] for i in range(len(processed_data))]
        prepared_data = RecordsBucket(filtered_data, self.available_time_res, self.main_timestep, self.window)
        # Step 3: Run strategy
        # Tell the broker what datetime it is, so it can mark trade orders to this timestamp
        self.broker.set_current_timestamp(timestep)
        self.strategy(prepared_data, timestep)

        # Step 4: Run broker
        current_data, next_tick_data, marginables, dividends, div_freq, short_rate, security_names = (
            self._prep_brokers_data(prepared_data.main.to_list()))
        self.broker.tick(timestep, next_time_step, security_names, current_data, next_tick_data, marginables, dividends, div_freq,
                         short_rate)

        # Step 5: Increase run_iter
        self.run_iter += 1

    @staticmethod
    def _prep_brokers_data(prepared_data: List[Record]) \
            -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[bool], npt.NDArray[np.float32],
            List[DividendFrequency], npt.NDArray[np.float32], List[str]]:
        """
        Prepare the data to feed the broker when it will do its tick.
        :param prepared_data: The main timestep data that was fed to the strategy
        :return: current data, the next tick data, the marginables, the dividends, the dividend frequency, the short rate
        and the security names
        """
        # Get security names
        security_names = [record.ticker for record in prepared_data]

        # Prepare current data for broker
        current_data = np.array([record.chart[["Open", "High", "Low", "Close"]].iloc[-1].to_list() for record in prepared_data], dtype=np.float32)
        next_tick_data = np.array([record.next_tick[["Open", "High", "Low", "Close"]].to_list() for record in prepared_data], dtype=np.float32)
        marginables = np.array([[record.marginable, record.shortable] for record in prepared_data], dtype=bool)
        dividends = np.array([record.chart["Dividends"].iloc[-1] if record.has_dividends else 0. for record in prepared_data], dtype=np.float32)
        div_freq = [record.div_freq for record in prepared_data]
        short_rate = np.array([record.short_rate for record in prepared_data], dtype=np.float32)

        return current_data, next_tick_data, marginables, dividends, div_freq, short_rate, security_names


    @staticmethod
    def _get_mask(main_data: List[Record]) -> npt.NDArray[bool]:
        """
        Get the mask to filter the data that are currently available.  (When Chart is not None)
        Where True means to keep it and False means to remove it.
        :param main_data: The main timestep data.
        :return: A boolean mask
        """
        mask = np.ones(len(main_data), dtype=bool)
        for idx, record in enumerate(main_data):
            if record.chart is None:
                mask[idx] = False    # Flip the mask to remove the record
        return mask

    def _prep_data(self, timestep: datetime) -> List[List[Record]]:
        """
        Prepare the data for the current timestep with a lookback period of size window.  The lookback period is the
        number of datapoint in the main timesteps.  It will be more for smaller resolution series and less for bigger
        ones.  This method assumes that nan are padding.  Nans means that the security didn't exist at the given
        timestep.  It is important to fill nan in data preprocessing steps before starting the backtest for in-data nans.
        :param timestep: The current backtest timestep
        :return: The prepared data
        """
        time_res = self.available_time_res[self.main_timestep]
        prepared_data: List[Optional[List[Record]]] = [None for _ in range(len(self.available_time_res))]

        # Prepare the data for main timestep
        prepared_data[self.main_timestep] = self._prepare_data(self._data, self.main_timestep, timestep, self.window,
                                                               self.default_marginable, self.default_shortable,
                                                               self.default_short_rate, save_next_tick=True)
        prepared_data[self.main_timestep] = self.apply_indicators(prepared_data[self.main_timestep],
                                                                  self.main_timestep, bigger_res=False)

        max_look_back_dt = datetime.fromisoformat('3000-01-01 00:00:00')
        for record in prepared_data[self.main_timestep]:
            if record.chart is not None:
                max_look_back_dt = min(max_look_back_dt, record.chart.index[0])

        # Check if there exists time resolution bigger than the main timesteps.  In that case, we need to crop the data.
        time_res_bigger_idx = [i for i, res in enumerate(self.available_time_res) if res > time_res]
        time_res_smaller_idx = [i for i, res in enumerate(self.available_time_res) if res < time_res]
        for idx in time_res_smaller_idx:
            prepared_data[idx] = self._prepare_data(self._data, idx, timestep + time_res - self.available_time_res[idx],
                                                    self.window,
                                                    self.default_marginable, self.default_shortable,
                                                    self.default_short_rate, max_look_back_dt=max_look_back_dt)
            prepared_data[idx] = self.apply_indicators(prepared_data[idx], idx, bigger_res=False)

        # Forge new candles for the one with bigger time resolution  to avoid peeking into the future.  (Data leaking)
        for idx in time_res_bigger_idx:
            series = self._prepare_data(self._data, idx, timestep, self.window,
                                        self.default_marginable, self.default_shortable,
                                        self.default_short_rate, max_look_back_dt=max_look_back_dt)
            last_candles = self.forge_last_candle(self._data, prepared_data, idx, timestep)
            for i, record in enumerate(series):
                if record.chart is not None:
                    record.chart.loc[record.chart.index[-1], ["Open", "High", "Low", "Close", "Volume", "Stock Splits"]] = \
                        pd.Series(last_candles[i][:-1], index=["Open", "High", "Low", "Close", "Volume", "Stock Splits"])

                    assert record.ticker == last_candles[i][-1], "An error happened, securities are no longer aligned"
            prepared_data[idx] = series

            # Apply indicators
            prepared_data[idx] = self.apply_indicators(prepared_data[idx], idx, bigger_res=True)
        return prepared_data

    def apply_indicators(self, data: List[Record], time_res_idx: int, bigger_res: bool = False) -> List[Record]:
        """
        Apply the indicators to the data (All tickers in time res)
        :param data: The list of records to apply the indicators to
        :param time_res_idx: The current time resolution
        :param bigger_res: Whether we are computing the indicator for a bigger time resolution or not.  If True and we
                    are in streaming mode, the last cache datapoint will be set to nan in order to recompute it.
        :return: The updated data
        """
        if isinstance(self.indicators, dict):
            indicators = self.indicators.get(time_res_idx)
            if indicators is None:
                return data
        elif isinstance(self.indicators, list):
            indicators = self.indicators[time_res_idx]
        else:
            indicators = self.indicators

        if len(indicators) > 0:
            for record in data:
                if record.chart is not None:
                    previous_data = self.cache_data[time_res_idx][record.ticker]
                    if previous_data is not None:
                        previous_data = previous_data.reindex(record.chart.index)
                    record.chart = self.run_indicator(record.chart, previous_data, indicators, bigger_res)
                    # Now cache the prepared indicators
                    if self.streaming_indicators:
                        self.cache_data[time_res_idx][record.ticker] = record.chart
        return data

    @staticmethod
    def run_indicator(data: pd.DataFrame, previous_data: Optional[pd.DataFrame],
                      indicators: IndicatorSet, streaming: bool,
                      bigger_res: bool = False) -> pd.DataFrame:
        """
        Run the indicators on a single chart.
        :param data: The chart data OHLCV
        :param previous_data: The previously cached data.  Used when streaming is True.  If streaming is False, this
                    parameter is ignored.  This should be a dataframe with the same index as the data.  This means that
                    the indicators points that needs to be calculated should be nan, and one already calculated should be
                    the previously calculated values.  The columns names should be the output columns names of the
                    indicators.
        :param indicators: The indicatorSet to use.
        :param streaming: Whether to use the streaming capabilities of the indicators or not.
        :param bigger_res: Whether we are computing the indicator for a bigger time resolution or not.  If True and we
                    are in streaming mode, the last cache datapoint will be set to nan in order to recompute it.
        :return: The new data
        """
        if not streaming:
            return indicators.run_all(data)
        else:
            if bigger_res:
                # We are going to set to nan the last row of the indicators.
                feat = indicators.out
                previous_data[feat].iloc[-1] = np.nan
            return indicators.run_all(data, previous_data)

    def run(self) -> BackTestResult:
        """
        Run the backtest simulation with the given data and strategy.
        It will show a progress bar showing the current progress.
        It will save the results and all the statistics into a BackTestResult object in the results attribute, or
        returned value of this method.
        :return: BackTestResult object containing all the results, statistics, debug info, etc.
        """
        start = datetime.now()
        # Step 1: Initialization
        # Initialize the backtest by checking if the data is valid and preparing the timestep list.  It also gets the
        # columns names (features) and the tickers names.
        features, tickers, timesteps_list = self._initialize_bcktst()

        # Initialize strategy
        self.strategy.init(self.account, self.broker, self.available_time_res)
        # Initialize cash controller
        self.cash_controller.init(self.account, self.broker, self.strategy)

        # This adds freedom to the user if he has custom data
        timesteps_list = self.stadardize_timesteps(timesteps_list)
        assert len(timesteps_list) > 0, "There is no data to backtest."

        # Initialize metadata with dynamic parameters
        self.metadata.init(self.strategy, backtest_parameters=self._backtest_parameters, tickers=tickers,
                           features=features)

        if self.metadata.time_res is None:
            self.metadata.time_res = self.available_time_res[self.main_timestep].total_seconds()

        # Initialize the cache
        self.cache_data = [{ticker: None for ticker in self._data[time_res]}
                           for time_res in range(len(self.available_time_res))]

        # Step 2: Run the backtest
        last_timestep: Optional[datetime] = None
        i: int = 0
        for i, timestep in enumerate(tqdm(timesteps_list[self.window:-1], desc="Backtesting...")):
            # Run the cash controller
            if last_timestep is None:
                last_timestep = timestep
            else:
                if timestep.day != last_timestep.day:
                    self.cash_controller.deposit(timestep, CashControllerTimeframe.DAY)
                if timestep.date().isocalendar()[1] != last_timestep.date().isocalendar()[1]:
                    self.cash_controller.deposit(timestep, CashControllerTimeframe.WEEK)
                if timestep.month != last_timestep.month:
                    self.cash_controller.deposit(timestep, CashControllerTimeframe.MONTH)
                if timestep.year != last_timestep.year:
                    self.cash_controller.deposit(timestep, CashControllerTimeframe.YEAR)
                last_timestep = timestep

            self.step(i, timestep, timesteps_list[i + self.window + 1])
            if self.broker.message.bankruptcy:
                print(f"\n\n\033[38;5;203mTHE STRATEGY WENT BANKRUPT!\033[0m\n")
                break

        # Get last timestep, it could be different if the strategy went bankrupt
        if self.broker.message.bankruptcy:
            last_timestep = timesteps_list[i + self.window + 1]
        else:
            last_timestep = timesteps_list[-1]

        if self.sell_at_the_end:
            self._sell_all(last_timestep)
            # We do this because selling all positions might add a margin call of type missing_funds.
            if "missing_funds" in self.broker.message.margin_calls:
                self.broker.message.bankruptcy_amount = self.broker.message.margin_calls["missing_funds"].amount


        # Step 3: Prepare and save stats
        if self.market_index is not None:
            market_worth = self.market_index.data["Close"].loc[timesteps_list[self.window]:timesteps_list[-1]].to_numpy()
            market_worth[0] = self.market_index.data["Open"].loc[timesteps_list[self.window]]
        else:
            market_worth = None

        self.results = BackTestResult(self.metadata.strategy_name,
                                      metadata=self.metadata,
                                      start=timesteps_list[self.window],
                                      end=last_timestep,
                                      intial_cash=self._initial_cash,
                                      added_cash=self.cash_controller._total_deposited,
                                      market_index=market_worth,
                                      broker=self.broker,
                                      account=self.account,
                                      risk_free_rate=self.risk_free_rate)
        end = datetime.now()
        run_duration = (end - start).total_seconds()

        # Now we can set the duration (After everything has been computed)
        self.results.metadata.run_duration = run_duration
        return self.results

    def _initialize_bcktst(self) -> Tuple[List[str], List[str], List[datetime]]:
        """
        Initialize the backtest by checking if the data is valid and preparing the timestep list.  It also gets the
        columns names (features) and the tickers names.
        :return: features, the tickers and the timesteps list
        """
        # For metadata
        features: List[str] = []
        tickers: List[str] = []
        timesteps_list: List[datetime] = []
        # Evaluate if data is valid + make timestep list.
        available_time_res: List[timedelta] = []

        # Add dynamically calculated time resolutions
        if self.time_res_extender is not None:
            self._data += self.time_res_extender.extend(self._data)

        # Many groups of time series
        for i, ts_group in enumerate(self._data):
            available_time_res.append(ts_group[list(ts_group.keys())[0]].time_res)
            # Check if all the data has the same time resolution + renorm data
            last_index: Optional[pd.Index] = None
            for ticker, ts in ts_group.items():
                if ts.time_res != available_time_res[i]:
                    raise ValueError(f"All the timeseries data must have the same time resolution.\n"
                                     f"Ticker: {ticker} for data in group {i} has a different time resolution than the "
                                     f"other data in the same group.")
                if last_index is None:
                    last_index = ts.data.index
                else:
                    if (last_index != ts.data.index).any():
                        raise ValueError(f"All the timeseries data in the same group must have the same index.\n"
                                            f"Ticker: {ticker} for data in group {i} has a different index than the "
                                            f"other data in the same group.")

                # Renormalize data by undoing splits:
                ts.data = self._reverse_split_norm(ts.data)

                if i == self.main_timestep:
                    if not features:
                        features = list(ts.data.columns)
                    if not tickers:
                        tickers = list(ts_group.keys())
                    if len(ts.data) > len(timesteps_list):
                        timesteps_list = list(ts.data.index)

        self.available_time_res = available_time_res

        return features, tickers, timesteps_list

    def _sell_all(self, timestep: datetime):
        self.broker.set_current_timestamp(timestep)

        # Prepare the data for main timestep
        prepared_data = self._prepare_data(self._data, self.main_timestep, timestep, window=self.window,
                                           default_marginable=self.default_marginable,
                                           default_shortable=self.default_shortable,
                                           default_short_rate=self.default_short_rate, save_next_tick=True,
                                           next_tick_is_current=True)
        current_data, next_tick_data, marginables, dividends, div_freq, short_rate, security_names = (
            self._prep_brokers_data(prepared_data))
        dic = {record.ticker: record for record in prepared_data}
        # Now, sell everything at market price
        long = self.broker.portfolio.getLong()
        for ticker, position in long.items():
            if position.amount > 0:
                close_price = dic[ticker].chart["Close"].iloc[-1]
                self.broker.sell_long(ticker, position.amount, None, (close_price, None))

        short = self.broker.portfolio.getShort()
        for ticker, position in short.items():
            if position.amount > 0:
                close_price = dic[ticker].chart["Close"].iloc[-1]
                self.broker.buy_short(ticker, position.amount, None, (None, close_price))

        self.broker.tick(timestep, timestep, security_names, current_data, next_tick_data, marginables, dividends,
                         div_freq, short_rate, last_tick=True)

    @staticmethod
    def _prepare_data(data: List[Dict[str, TSData]], current_time_res: int, timestep: datetime,
                      window: int, default_marginable: bool, default_shortable: bool, default_short_rate: float,
                      max_look_back_dt: Optional[datetime] = None, save_next_tick: bool = False,
                      next_tick_is_current: bool = False) -> List[Record]:
        """
        Prepare the data for the current timestep.  This method assumes that nan are padding.  This means that it is
        assumed that the security didn't exist at the time where there is nan.  It is important to fill nan in data
        preprocessing steps before starting the backtest.  If you do not want to impute value, you can override this
        method.  This method will also handle splits by dividing the price by split value and multiplying the volume by
        the split value.  (This is the default behavior of yfinance)
        If the security does not exist yet, it will return a Record with None as chart and None as a next tick.
        :param data: The data
        :param current_time_res: The current time resolution
        :param timestep: The current timestep
        :param window: The window size (How long in the past the strategy can see)
        :param default_marginable: The default value for marginable if it is not in the data
        :param default_shortable: The default value for shortable if it is not in the data
        :param default_short_rate: The default value for short rate if it is not in the data
        :param max_look_back_dt: The maximum look back timestep.  If not None, the window will start at that timestep
                                or the nearest one before that timestep in the data's index assuming it is not padded
                                with nan.  This can be useful for time resolutions that are not the main one.
        :param save_next_tick: If True, it will save the current tick data as a next tick in the Record object.
                                (This is useful for the sell all at the end of the backtest)
        :param next_tick_is_current: If True, the next tick will be the current tick.  This is useful at the end of the
                                        backtest to sell everything at current market value.
        :return: The prepared data
        """
        prepared_data: List[Record] = []
        for ticker, ts in data[current_time_res].items():
            cropped = ts.data.loc[:timestep]
            timestep = cropped.iloc[-1].name
            timestep_idx = ts.data.index.get_loc(timestep) + 1

            # Find start padding
            start_idx = np.argmin(np.isnan(cropped[["Open", "High", "Low", "Close", "Volume"]].to_numpy()).any(axis=1))

            # Find end padding
            if np.isnan(cropped[["Open", "High", "Low", "Close", "Volume"]].to_numpy()[-1]).any():
                end_idx = start_idx + np.argmax(
                    np.isnan(cropped[["Open", "High", "Low", "Close", "Volume"]].to_numpy()[start_idx:]).any(axis=1))
            else:
                end_idx = timestep_idx

            # The security does not exist yet
            if end_idx == start_idx:
                prepared_data.append(Record(None, ticker, current_time_res, False, False,
                                            ts.div_freq, 0))
                continue

            # Security has been delisted or there is missing values.  (We will ignore it)
            if end_idx != timestep_idx:
                prepared_data.append(Record(None, ticker, current_time_res, False, False,
                                            ts.div_freq, 0, None))
                continue

            if max_look_back_dt is not None:
                start_idx = ts.data.index.get_loc(ts.data.loc[:max_look_back_dt].iloc[-1].name)

            # Check if window is too big
            if end_idx - start_idx > window and max_look_back_dt is None:
                start_idx = end_idx - window

            if "Marginable" in ts.data.columns:
                marginable = ts.data["Marginable"].iloc[end_idx - 1]
            else:
                marginable = default_marginable
            if "Shortable" in ts.data.columns:
                shortable = ts.data["Shortable"].iloc[end_idx - 1]
            else:
                shortable = default_shortable
            if "Short_rate" in ts.data.columns:
                short_rate = ts.data["Short_rate"].iloc[end_idx - 1]
            else:
                short_rate = default_short_rate

            # Normalize the price and volume of window according to splits
            multiplier = cropped["Stock Splits"].max()
            cropped = cropped.iloc[start_idx:]
            if multiplier > 0:
                cropped["Open"] /= multiplier
                cropped["High"] /= multiplier
                cropped["Low"] /= multiplier
                cropped["Close"] /= multiplier
                cropped["Volume"] *= multiplier
                cropped["Dividends"] *= multiplier
            if save_next_tick:
                if next_tick_is_current:
                    end_idx = end_idx - 1
                next_point = ts.data[["Open", "High", "Low", "Close", "Volume", "Dividends"]].iloc[end_idx]
                if multiplier > 0:
                    next_point["Open"] /= multiplier
                    next_point["High"] /= multiplier
                    next_point["Low"] /= multiplier
                    next_point["Close"] /= multiplier
                    next_point["Volume"] *= multiplier
                    next_point["Dividends"] *= multiplier
                if ts.data["Stock Splits"].iloc[end_idx] > 0:
                    next_point[["Open", "High", "Low", "Close"]] /= ts.data["Stock Splits"].iloc[end_idx]
                    next_point[["Volume", "Dividends"]] *= ts.data["Stock Splits"].iloc[end_idx]
                prepared_data.append(Record(cropped, ticker, current_time_res,
                                        marginable, shortable, ts.div_freq, short_rate, next_point))
            else:
                prepared_data.append(Record(cropped, ticker, current_time_res,
                                        marginable, shortable, ts.div_freq, short_rate, None))



        return prepared_data

    @staticmethod
    def stadardize_timesteps(timesteps_list: List[datetime]) -> List[datetime]:
        """
        Optionnaly reformat timesteps_list.  Can be useful when indexes are variable across stocks
        :param timesteps_list: The list of timesteps
        :return: The reformatted list of timesteps
        """
        return timesteps_list


    def forge_last_candle(self, data: List[Dict[str, TSData]], prepared_data: List[List[Record]], current_time_res: int,
                      timestep: datetime) -> List[Tuple[float, float, float, float, float, float, str]]:
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
        :return: A list of the forged candles: List[Tuple[Open, High, Low, Close, Volume, Stock Split, ticker]]
        """
        if self._verbose >= 2 and self.run_iter == 0:
            warnings.warn("This method is not guaranteed to work for your setup.  You should override it, or make sure it "
                          "works for your setup if you have series that have a higher resolution than the main resolution.",
                          UnexpectedBehaviorRisk)
        return self.default_forge_last_candle(data, current_time_res, timestep, self.main_timestep)

    @staticmethod
    def default_forge_last_candle(data: List[Dict[str, TSData]], current_time_res: int,
                      timestep: datetime, main_timestep: int) -> List[Tuple[float, float, float, float, float, float, str]]:
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
        :param current_time_res: The current time resolution (to forge)
        :param timestep: The current timestep
        :param main_timestep: The main timestep
        :return: A list of the forged candles: List[Tuple[Open, High, Low, Close, Volume, Stock Split, ticker]]
        """
        newly_forged_candles = []
        for ticker, ts in data[current_time_res].items():
            # Step 1: Find the last candle compared to current time step with the current resolution. (Get index and Open)
            last_candle = ts.data.loc[:timestep].iloc[-1]
            candle_open = last_candle["Open"]
            start_index = last_candle.name

            # Step 2: Find the last candle for the ticker with main resolution.  (Get index and Close)
            main_res_data = data[main_timestep][ticker].data
            main_res_last_candle = main_res_data.loc[:timestep].iloc[-1]
            candle_close = main_res_last_candle["Close"]
            end_index = main_res_last_candle.name

            # Step 3: Get the series between the beginning of the candle (Current resolution) and the current timestep.
            intra_candle = main_res_data.loc[start_index:end_index]

            # Step 4: Get missing OHLC stats: High, Low and Volume
            candle_high = intra_candle["High"].max()
            candle_low = intra_candle["Low"].min()
            candle_volume = intra_candle["Volume"].sum()
            splits_df = intra_candle["Stock Splits"].to_numpy()
            splits_df[splits_df == 0] = 1
            splits = splits_df.prod()
            if splits == 1:
                splits = 0

            # Step 5: Add this to new forged candles
            newly_forged_candles.append((candle_open, candle_high, candle_low, candle_close, candle_volume, splits, ticker))

        return newly_forged_candles

    @staticmethod
    def _reverse_split_norm(hist: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the pricing and volume by undoing the splits far a single timeseries.
        It also updates the Stock Splits column to be the current multiplier.  (yfinance price is price  / multiplier)
        This means that it starts with a multiplier of 1 and it is mulitplied by the split value each time there is a
        split.
        Warning:
            This is a mutating function.  It will modify the dataframe in place.
        :param hist: An OHLCV dataframe
        :return: The modified dataframe
        """

        class Normalizer:
            def __init__(self):
                self.multiplier = 1

            def __call__(self, row):
                if row["Stock Splits"] > 0:
                    self.multiplier *= row["Stock Splits"]

                new_row = row.copy()
                new_row["Stock Splits"] = self.multiplier
                return new_row

        idx_splits = np.arange(len(hist))[(hist["Stock Splits"] > 0)]
        multipliers: np.ndarray = hist["Stock Splits"].to_numpy()[idx_splits]
        total_multiplier = multipliers.prod()
        hist["Open"] *= total_multiplier
        hist["High"] *= total_multiplier
        hist["Low"] *= total_multiplier
        hist["Close"] *= total_multiplier
        hist["Volume"] = np.round(hist["Volume"] / total_multiplier)
        if "Dividends" in hist.columns:
            hist["Dividends"] = (hist["Dividends"] / total_multiplier)
        hist.apply(Normalizer(), axis=1)
        return hist
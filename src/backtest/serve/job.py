from datetime import datetime, timedelta
from ..engine import Broker, Account, Strategy, Backtest, BackTestResult, CashControllerBase, TimeResExtender, TradeOrder
from ..engine import BuyLongOrder, SellLongOrder, BuyShortOrder, SellShortOrder, Record, TSData, CashControllerTimeframe
from ..engine import RecordsBucket, DividendFrequency
from ..data import DataPipe
from ..indicators import IndicatorSet
from typing import Any, Optional, List, Dict, Union, Tuple, Callable
from pathlib import PurePath
import os
import json
import numpy as np
import numpy.typing as npt
from .state_signals import StateSignals
from .renderer import Renderer, RendererList


class RecordingBroker(Broker):
    """
    A broker that records the signals given by the strategy.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal: Dict[str, TradeOrder] = {}
    def bind(self, signal: Dict[str, TradeOrder]):
        self._signal = signal

    def buy_long(self, ticker: str, amount: int, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        super().buy_long(ticker, amount, amount_borrowed, expiry, price_limit)
        self._signal[ticker] = BuyLongOrder(self._current_timestamp, ticker, price_limit, amount, amount_borrowed,
                                            expiry)

    def sell_long(self, ticker: str, amount: int, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        super().sell_long(ticker, amount, expiry, price_limit)
        self._signal[ticker] = SellLongOrder(self._current_timestamp, ticker, price_limit, amount, 0,
                                             expiry)
    def buy_short(self, ticker: str, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        super().buy_short(ticker, amount_borrowed, expiry, price_limit)
        self._signal[ticker] = BuyShortOrder(self._current_timestamp, ticker, price_limit, 0, amount_borrowed,
                                                expiry)

    def sell_short(self, ticker: str, amount_borrowed: int = 0, expiry: Optional[datetime] = None,
                 price_limit: Tuple[Optional[float], Optional[float]] = (None, None)):
        super().sell_short(ticker, amount_borrowed, expiry, price_limit)
        self._signal[ticker] = SellShortOrder(self._current_timestamp, ticker, price_limit, 0, amount_borrowed,
                                                expiry)


class Job(Backtest):
    def __init__(self, strategy: Strategy, data: DataPipe, lookback: timedelta, result_path: str, *,
                 index_pipe: Optional[DataPipe] = None,
                 working_directory: PurePath = PurePath("./prod_data"),
                 initial_cash: float = 100000,
                 indicators: Union[IndicatorSet, List[IndicatorSet], Dict[int, IndicatorSet]] = IndicatorSet(),
                 trigger_cb: Optional[Callable[[StateSignals, PurePath], None]] = None,
                 renderer: Union[RendererList, Renderer] = None, cash_controller: CashControllerBase = CashControllerBase(),
                 time_res_extender: Optional[TimeResExtender] = None):
        self.data_pipe = data
        self.lookback = lookback
        self._trigger_cb = trigger_cb
        self._renderer = renderer
        results = BackTestResult.load(result_path)
        params = results.metadata.backtest_parameters
        super().__init__([], strategy, main_timestep=params['main_timestep'], initial_cash=initial_cash,
                         commission=params['commission'], relative_commission=params['relative_commission'],
                         margin_interest=params['margin_interest'], min_initial_margin=params['min_initial_margin'],
                         min_maintenance_margin=params['min_maintenance_margin'],
                         liquidation_delay=params['liquidation_delay'],
                         min_initial_margin_short=params['min_initial_margin_short'],
                         min_maintenance_margin_short=params['min_maintenance_margin_short'],
                         broker=RecordingBroker, account=Account, window=params['window'],
                         default_marginable=params['default_marginable'], default_shortable=params['default_shortable'],
                         default_short_rate=params['default_short_rate'],
                         risk_free_rate=params['risk_free_rate'], sell_at_the_end=False,
                         cash_controller=cash_controller, verbose=params['verbose'],
                         time_res_extender=time_res_extender, indicators=indicators)
        self.signal = {}
        self.broker.bind(self.signal)
        self.working_directory = working_directory
        self.last_timestep: Optional[datetime] = None
        self.index_pipe = index_pipe
        self._index_data: Optional[List[Dict[str, TSData]]] = None

    def run(self) -> BackTestResult:
        raise NotImplementedError("This method is not implemented for Job, use Backtest to run run_jib instead.")

    def run_job(self, every):
        # TODO: Handle the every parameter if not None
        self.pipeline()

    def setup(self):
        """
        This method will setup the backtest.  It will load from cache, if it exists, the backtest state.
        :return: None
        """
        # Initialize the dynamically extended time resolution and get the available time resolutions
        _, _, timestep_list = self._initialize_bcktst()
        # In case the class is derived and the method standardize_timesteps is overridden
        timesteps_list = self.stadardize_timesteps(timestep_list)
        assert len(timesteps_list) > 0, "There is no data to backtest."

        # Initialize the cache, even though it won't be used, it is necessary to avoid errors
        self.cache_data = [{ticker: None for ticker in self._data[time_res]}
                           for time_res in range(len(self.available_time_res))]

        # Now that the setup is done, we try to load the state from the cache.
        if os.path.exists(self.working_directory / PurePath("cache/job_cache.json")):
            with open(self.working_directory / PurePath("cache/job_cache.json"), "r") as f:
                data = json.load(f)
            self.account = Account.load_state(data["account"])
            self.broker = RecordingBroker.load_state(data["broker"], self.account)
            self.last_timestep = datetime.fromisoformat(data["last_timestep"])
            self.strategy = self.strategy.load(self.working_directory / PurePath("cache/strategy.pkl"))
            self.broker.bind(self.signal)

    def pipeline(self):
        """
        This method will run the whole pipeline.
        :return: None
        """
        # Step 1: Fetch the data
        now = datetime.now()
        self._data: List[Dict[str, TSData]] = self.data_pipe.get(now - self.lookback, now)
        if self.index_pipe is not None:
            self._index_data = self.index_pipe.get(now - self.lookback, now)

        # Step 2: Setup the object
        self.setup()
        self.strategy.init(self.account, self.broker, self.available_time_res)

        # Step 3: Run the cash controller (if in the right conditions)
        if self.last_timestep is not None:
            if now.day != self.last_timestep.day:
                self.cash_controller.deposit(now, CashControllerTimeframe.DAY)
            if now.date().isocalendar()[1] != self.last_timestep.date().isocalendar()[1]:
                self.cash_controller.deposit(now, CashControllerTimeframe.WEEK)
            if now.month != self.last_timestep.month:
                self.cash_controller.deposit(now, CashControllerTimeframe.MONTH)
            if now.year != self.last_timestep.year:
                self.cash_controller.deposit(now, CashControllerTimeframe.YEAR)

        # Step 4: Prepare the data, no need to filter None charts, as the data is supposed to be up-to-date
        processed_data: List[List[Record]] = self._prep_data(now)
        prepared_data = RecordsBucket(processed_data, self.available_time_res, self.main_timestep, self.window)

        # Step 5: If the cache is not None, (The strategy ran before), we run the broker to keep track of the current
        # performances
        if os.path.exists(self.working_directory / PurePath("cache/job_cache.json")):
            broker_data = self._prep_data(self.last_timestep)
            broker_data = RecordsBucket(broker_data, self.available_time_res, self.main_timestep, self.window)
            current_data, next_tick_data, marginables, dividends, div_freq, short_rate, security_names = (
                self._prep_brokers_data(broker_data.main.to_list()))
            self.broker.tick(self.last_timestep, now, security_names, current_data, next_tick_data, marginables,
                             dividends, div_freq,
                             short_rate)

        self.last_timestep = now

        # Step 6: Run the strategy
        self.broker.set_current_timestamp(now)
        self.strategy(prepared_data, now)

        # Step 7: Save the state
        with open(self.working_directory / PurePath("cache/job_cache.json"), "w") as f:
            json.dump({"account": self.account.save_state(), "broker": self.broker.save_state(),
                       "last_timestep": now.isoformat()}, f)
        self.strategy.save(self.working_directory / PurePath("cache/strategy.pkl"))

        # Step 8: Package the signals and the current state in a ActionStates object
        state_signals = StateSignals(self.account, self.broker, self.signal, now, self._index_data)

        # Step 9: Render the report using the renderer
        if self._renderer is not None:
            self._renderer.render(state_signals, base_path=self.working_directory / PurePath("reports"))

        # Step 10: Call the trigger callback
        if self._trigger_cb is not None:
            self._trigger_cb(state_signals, self.working_directory)


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
        yesterday_data = np.array([record.chart[["Open", "High", "Low", "Close"]].iloc[-2].to_list() for record in prepared_data], dtype=np.float32)
        current_tick_data = np.array([record.chart[["Open", "High", "Low", "Close"]].iloc[-1].to_list() for record in prepared_data], dtype=np.float32)
        marginables = np.array([[record.marginable, record.shortable] for record in prepared_data], dtype=bool)
        dividends = np.array([record.chart["Dividends"].iloc[-2] if record.has_dividends else 0. for record in prepared_data], dtype=np.float32)
        div_freq = [record.div_freq for record in prepared_data]
        short_rate = np.array([record.short_rate for record in prepared_data], dtype=np.float32)

        return yesterday_data, current_tick_data, marginables, dividends, div_freq, short_rate, security_names
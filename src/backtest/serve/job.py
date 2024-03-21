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
import time
from .state_signals import StateSignals, ServerStatus
from .renderer import Renderer, RendererList
import traceback
import warnings

try:
    import schedule
    SCHEDULE_INSTALLED = True
except ImportError:
    SCHEDULE_INSTALLED = False


class Job(Backtest):
    def __init__(self, strategy: Strategy, data: DataPipe, lookback: timedelta, *, result_path: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None, index_pipe: Optional[DataPipe] = None,
                 working_directory: PurePath = PurePath("./prod_data"),
                 indicators: Union[IndicatorSet, List[IndicatorSet], Dict[int, IndicatorSet]] = IndicatorSet(),
                 trigger_cb: Optional[Callable[[StateSignals, PurePath], None]] = None,
                 renderer: Union[RendererList, Renderer] = None, cash_controller: CashControllerBase = CashControllerBase(),
                 time_res_extender: Optional[TimeResExtender] = None):
        if result_path is None and params is None:
            raise ValueError("You must provide either a result_path or a params dictionary")
        if result_path is not None:
            results = BackTestResult.load(result_path)
            params = results.metadata.backtest_parameters
        self.params = params
        self.data_pipe = data
        self.lookback = lookback
        self._trigger_cb = trigger_cb
        self._renderer = renderer
        super().__init__([], strategy, main_timestep=params['main_timestep'], initial_cash=params['initial_cash'],
                         commission=params['commission'], relative_commission=params['relative_commission'],
                         margin_interest=params['margin_interest'], min_initial_margin=params['min_initial_margin'],
                         min_maintenance_margin=params['min_maintenance_margin'],
                         liquidation_delay=params['liquidation_delay'],
                         min_initial_margin_short=params['min_initial_margin_short'],
                         min_maintenance_margin_short=params['min_maintenance_margin_short'],
                         broker=Broker, account=Account, window=params['window'],
                         default_marginable=params['default_marginable'], default_shortable=params['default_shortable'],
                         default_short_rate=params['default_short_rate'],
                         risk_free_rate=params['risk_free_rate'], sell_at_the_end=False,
                         cash_controller=cash_controller, verbose=params['verbose'],
                         time_res_extender=time_res_extender, indicators=indicators)
        self.working_directory = working_directory
        self.last_timestep: Optional[datetime] = None
        self.index_pipe = index_pipe
        self._index_data: Optional[List[TSData]] = None

    def run(self) -> BackTestResult:
        raise NotImplementedError("This method is not implemented for Job, use Backtest to run run_jib instead.")

    def run_job(self, every: Optional['schedule.Job'] = None):
        if SCHEDULE_INSTALLED and every is not None:
            every.do(self.pipeline)
            while True:
                schedule.run_pending()
                time.sleep(1)

        else:
            self.pipeline()

    def setup(self) -> Optional[datetime]:
        """
        This method will setup the backtest.  It will load from cache, if it exists, the backtest state.
        """
        # Initialize the dynamically extended time resolution and get the available time resolutions
        self._initialize_bcktst()

        # Initialize the cache, even though it won't be used, it is necessary to avoid errors
        self.cache_data = [{ticker: None for ticker in self._data[time_res]}
                           for time_res in range(len(self.available_time_res))]

        prev_last_data_dt = None
        # Now that the setup is done, we try to load the state from the cache.
        if os.path.exists(self.working_directory / PurePath("cache/job_cache.json")):
            with open(self.working_directory / PurePath("cache/job_cache.json"), "r") as f:
                data = json.load(f)
            self.account = Account.load_state(data["account"])
            self.broker = Broker.load_state(data["broker"], self.account)
            self.last_timestep = datetime.fromisoformat(data["last_timestep"])
            prev_last_data_dt = datetime.fromisoformat(data["last_data_dt"])
            self.strategy = self.strategy.load(self.working_directory / PurePath("cache/strategy.pkl"))

        return prev_last_data_dt

    def pipeline(self, now_override: Optional[datetime] = None):
        """
        This method will run the whole pipeline.
        :param now_override: A datetime object to override the current time.  (Useful for testing in simulated environments)
        :return: None
        """
        error = False
        warning = False
        try:
            with warnings.catch_warnings(record=True) as w:
                # Step 1: Fetch the data
                now = datetime.now() if now_override is None else now_override
                self._data: List[Dict[str, TSData]] = self.data_pipe.get(now - self.lookback, now)
                if self.index_pipe is not None:
                    self._index_data = self.index_pipe.get(now - self.lookback, now)

                # Step 2: Setup the object
                previous_last_data_dt = self.setup()
                self.strategy.init(self.account, self.broker, self.available_time_res)
                self.cash_controller.init(self.account, self.broker, self.strategy)

                # Step 4: Prepare the data, no need to filter None charts, as the data is supposed to be up-to-date
                processed_data: List[List[Record]] = self._prep_data(now)
                prepared_data = RecordsBucket(processed_data, self.available_time_res, self.main_timestep, self.window)
                last_data_dt = prepared_data.main[0].chart.index[-1]

                # If this condition is true, it means that the market was closed, the data wasn't updated, so we must not run
                # anything and return
                if previous_last_data_dt is not None:
                    if last_data_dt == previous_last_data_dt:
                        return
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

                # Step 6: Run the strategy
                self.broker.set_current_timestamp(now)
                self.strategy(prepared_data, now)

                # Step 7: Save the state
                if not os.path.exists(self.working_directory / PurePath("cache")):
                    os.makedirs(self.working_directory / PurePath("cache"))
                with open(self.working_directory / PurePath("cache/job_cache.json"), "w") as f:
                    json.dump({"account": self.account.get_state(), "broker": self.broker.get_state(),
                               "last_timestep": now.isoformat(), "last_data_dt": last_data_dt.isoformat()}, f)
                self.strategy.save(self.working_directory / PurePath("cache/strategy.pkl"))

            # Check if warnings were raised
            if len(w) > 0:
                warning = True
                for warn in w:
                    warnings.warn(warn.message, warn.category)
        except Exception as e:
            error = True
            traceback.print_exc()
            exception = e

        # Step 8: Package the signals and the current state in a ActionStates object
        if error:
            status = ServerStatus.ERROR
        elif warning:
            status = ServerStatus.WARNING
            exception = None
        else:
            status = ServerStatus.OK
            exception = None
        signal = {order.security: order for order in self.broker.pending_orders}
        state_signals = StateSignals(self.account, self.broker, signal, self.strategy, now, self.cash_controller,
                                     self._initial_cash, self._index_data, self._data, self.main_timestep,
                                     self.params, status, exception=exception, warnings=w)

        # Step 9: Render the report using the renderer
        if self._renderer is not None:
            self._renderer.render(state_signals, base_path=self.working_directory / PurePath("reports"))

        # Step 10: Call the trigger callback
        if self._trigger_cb is not None:
            self._trigger_cb(state_signals, self.working_directory)

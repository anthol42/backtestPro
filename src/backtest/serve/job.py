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
from datetime import datetime, timedelta
from ..engine import Broker, Account, Strategy, Backtest, BackTestResult, CashControllerBase, TimeResExtender
from ..engine import Record, TSData, CashControllerTimeframe
from ..engine import RecordsBucket
from ..data import DataPipe
from ..indicators import IndicatorSet
from typing import Any, Optional, List, Dict, Union, Callable
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
    """
    This class is designed to run a strategy in inference mode.  It is derived from the backtest class to keep as
    much similarity as possible.  The main difference is that the Job class will not run the strategy in a loop, but
    will run it once and return the signals.  It is designed to be used in a server environment where the strategy
    will be run on a schedule.  (Cron schedule or with the schedule module)

    Overall mechanism:
    The Job class receives as parameter a datapipe.  When the job class is run, it starts by fetching the data from the
    datapipe.  It then prepares the data as it is prepared in the backtest class.  It then runs the strategy on the
    data and records the signals emitted by the strategy.  It builds a StateSignals object with the signals and the
    state of the system (i.e. the account, the broker, the cash controller, etc.).  It then calls the renderer.  The
    renderer converts the StateSignals object into a file (or a set of files) that can be used to visualize the signals
    or be imported into an application.  Finally, the Job class calls a trigger callback that can be used to trigger
    other actions such as sending a notification, buying or selling assets based on an api, etc.

    How to use:
    The Job class is designed to be run once a backtest has been completed (To see the performance expected by the
    strategy).  Also, there are two ways to run the Job class.  The first way is to call run the run_job method once
    in a while (Could be scheduled with a Cron job).  The second way, is to pass a schedule.Job object to the run_job
    method and the script will run the job at the specified interval.  In that case, the script never finishes.

    Example:
        The first example is designed to be run by a Cron job.  The script will brun and exit.
        >>> # imports ...
        >>> data_pipe = ...  # A data pipe that fetches the data
        >>> strategy = ...  # A strategy object
        >>> # In case a backtest has been run prior to this, we pass the path to the backtest results.  This way, the
        >>> # Job class will load the initial parameters from the backtest results.
        >>> job = Job(strategy, data_pipe, timedelta(days=1), result_path="path/to/backtest/result", renderer=...,
        ...          trigger_cb=...)
        >>> # In case a backtest hasn't been run, we pass the parameters directly to the Job class.
        >>> # The parameters are the parameters passed to the backtest class.  Almost every parameters must be passed,
        >>> # because it doesn't have defaults.
        >>> job = Job(strategy, data_pipe, timedelta(days=1), params={"initial_cash": 10000, "commission": 0.01, ...},
        ...          renderer=..., trigger_cb=...)
        >>> job.run_job()

        The second example is designed to be run in a loop.  The script will run the job every day at 5pm.
        >>> # imports ...
        >>> data_pipe = ...  # A data pipe that fetches the data
        >>> strategy = ...  # A strategy object
        >>> # In case a backtest has been run prior to this, we pass the path to
        >>> # the backtest results.  This way, the Job class will load the initial parameters from the backtest results.
        >>> job = Job(strategy, data_pipe, timedelta(days=1), result_path="path/to/backtest/result", renderer=...,
        ...          trigger_cb=...)
        >>> # In case a backtest hasn't been run, we pass the parameters directly to the Job class.
        >>> # The parameters are the parameters passed to the backtest class.  Almost every parameters must be passed,
        >>> # because it doesn't have defaults.
        >>> job = Job(strategy, data_pipe, timedelta(days=1), params={"initial_cash": 10000, "commission": 0.01, ...},
        ...          renderer=..., trigger_cb=...)
        >>> job.run_job(schedule.every().day.at("17:00"))
    """
    def __init__(self, strategy: Strategy, data: DataPipe, lookback: timedelta, *, result_path: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None, index_pipe: Optional[DataPipe] = None,
                 working_directory: PurePath = PurePath("./prod_data"),
                 indicators: Union[IndicatorSet, List[IndicatorSet], Dict[int, IndicatorSet]] = IndicatorSet(),
                 trigger_cb: Optional[Callable[[StateSignals, PurePath], None]] = None,
                 renderer: Union[RendererList, Renderer] = None, cash_controller: CashControllerBase = CashControllerBase(),
                 time_res_extender: Optional[TimeResExtender] = None):
        """
        :param strategy: The strategy to use and extract signals from.
        :param data: The data pipeline used to fetch the data for the strategy.  It must return a list of dict of TSData
        where position in the list are different time resolutions (like in the backtest) and the keys in the dict are
        the tickers.
        :param lookback: The lookback period to fetch the data.  For example, let's say the lookback is 1y, the data
        fetched will be from (now - 1y) to now at every run.
        :param result_path: The path to the backtest results.  If this is provided, the Job class will load the
        parameters from the backtest results.  If this is not provided, the Job class will load the parameters from the
        params dictionary.
        :param params: The parameters to use for the backtest.  If the result_path is provided, this parameter is
        ignored.  The parameters are the same as the parameters passed to the backtest class.  Almost every parameters
        must be passed, because it doesn't have defaults.  The parameters required are:
            - main_timestep: The main timestep to use for the backtest.
            - initial_cash: The initial cash to start the backtest.
            - commission: The commission to apply to the trades.
            - relative_commission: The relative commission to apply to the trades.
            - margin_interest: The margin interest to apply to the trades.
            - min_initial_margin: The minimum initial margin to apply to the trades.
            - min_maintenance_margin: The minimum maintenance margin to apply to the trades.
            - liquidation_delay: The liquidation delay to apply to the trades.
            - min_initial_margin_short: The minimum initial margin for short trades.
            - min_maintenance_margin_short: The minimum maintenance margin for short trades.
            - window: The window to use for the backtest.
            - default_marginable: The default marginable status used in the backtest.
            - default_shortable: The default shortable status used in the backtest.
            - default_short_rate: The default short rate used in the backtest.
            - risk_free_rate: The risk-free rate used in the backtest.
            - verbose: The verbosity level of the backtest.
        :param index_pipe: A datapipe designed to fetch the index data.  This is useful to compare the strategy
        performance with the index performance.  It must return a list of TSData objects.  Although it is recommended to
        use only one reference index because the renderers are designed to render only one index, it is possible to
        return more than one.  It is required that the time resolution of the index is the same as the main time
        resolution.
        :param working_directory: The working directory to use to export files.  This will be considered as the root by
        the renderers.  For example, the prebuilt renderers will export their reports in the reports directory of the
        working directory.
        :param indicators: The same indicators used for the backtest.
        :param trigger_cb: The trigger callback to call after the renderer has rendered the StateSignals.
        This callback is designed to trigger other actions like sending a notification, buying or selling assets based
        on an api, etc.
        :param renderer: A renderer object designed to convert the StateSignals object to an output file.  The renderer
        can be a single renderer or a list of renderers (RendererList).  The renderers are run in the same order as they
        are in the list.
        :param cash_controller: The cash controller used during the backtest.
        :param time_res_extender: The time resolution extender used during the backtest.
        """
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
        """
        This method will run the job.  If the every parameter is None, the job will run once and render the signals.
        If the every parameter is not None, the job will run at the specified interval and never return (never exit).
        :param every: The interval to run the job.
        :return: None
        """
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
        Loading from cache the backtest state simulate a warm restart, (like the script has never stopped).
        If the script is runned in a cron job, so it exits after each run, and restarts at the next run, it wouldn't be
        able to remember the state of the job and would think it is always the first run.  This is why the cache is
        used.  It is used to remember the state of the job between runs.  It is useful to remember the state to calulate
        the current statistics of the strategy.  It will later be possible to compare the current performance with the
        backtest performances.
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
        The steps of the pipeline are:
            1. Fetch the data
            2. Setup the Job.  (Initialize the different variables)
            3. Prepare the data
            4. Run the cash controller
            5. Run the broker  (It will always be run on the previous timestep, i.e. lagging by one timestep)
            6. Run the strategy
            7. Save the state of the job for later warm restarts.
            8. Package the signals and the state in a StateSignals object
            9. Render the signals using the renderer
            10. Call the trigger callback
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

                # Step 3: Prepare the data, no need to filter None charts, as the data is supposed to be up-to-date
                processed_data: List[List[Record]] = self._prep_data(now)
                prepared_data = RecordsBucket(processed_data, self.available_time_res, self.main_timestep, self.window)
                last_data_dt = prepared_data.main[0].chart.index[-1]

                # If this condition is true, it means that the market was closed, the data wasn't updated, so we must not run
                # anything and return
                if previous_last_data_dt is not None:
                    if last_data_dt == previous_last_data_dt:
                        return
                # Step 4: Run the cash controller (if in the right conditions)
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

        # Step 8: Package the signals and the current state in a StateSignals object
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

        # Step 9: Render the signals using the renderer
        if self._renderer is not None:
            self._renderer.render(state_signals, base_path=self.working_directory / PurePath("reports"))

        # Step 10: Call the trigger callback
        if self._trigger_cb is not None:
            self._trigger_cb(state_signals, self.working_directory)

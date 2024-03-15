from unittest import TestCase
from src.backtest.serve.job import Job, RecordingBroker
from src.backtest.engine import Account, TradeOrder, TradeType, TSData, Record, Strategy, RecordsBucket, Metadata
from src.backtest.data import Fetch, ToTSData, Process
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import PurePath
import os
import shutil


class TestRecordingJob(TestCase):
    def test_recording(self):
        """
        Test if recording signals is working
        """
        def myStrat(broker: RecordingBroker):
            broker.buy_long("AAPL", 100, 100, price_limit=(150, None))
            broker.sell_long("NVDA", 100, price_limit=(None, 150))
            broker.buy_short("TSLA", 200)
            broker.sell_short("AMZN", 200, price_limit=(200, None))

        # Initialization
        account = Account()
        broker = RecordingBroker(account)

        signals = {}
        broker.bind(signals)
        broker.set_current_timestamp(datetime(2021, 1, 1))

        # Run strategy tha create signals
        myStrat(broker)

        # Check if signals were recorded
        self.assertEqual(len(signals), 4)
        self.assertEqual(signals["AAPL"], TradeOrder(datetime(2021, 1, 1), "AAPL",
                                                     (150, None), 100, 100,
                                                     TradeType.BuyLong, None))
        self.assertEqual(signals["NVDA"], TradeOrder(datetime(2021, 1, 1), "NVDA",
                                                        (None, 150), 100, 0,
                                                        TradeType.SellLong, None))
        self.assertEqual(signals["TSLA"], TradeOrder(datetime(2021, 1, 1), "TSLA",
                                                        (None, None), 0, 200,
                                                        TradeType.BuyShort, None))

        self.assertEqual(signals["AMZN"], TradeOrder(datetime(2021, 1, 1), "AMZN",
                                                        (200, None), 0, 200,
                                                        TradeType.SellShort, None))
@Fetch
def LoadPipe(*args, **kwargs):
    """
    Load 1d charts
    """
    chart1 = pd.read_csv("../engine_tests/test_data/AAPL_6mo_1d.csv", index_col="Date")
    chart2 = pd.read_csv("../engine_tests/test_data/NVDA_6mo_1d.csv", index_col="Date")
    return {"AAPL": chart1, "NVDA": chart2}


class MyStrat(Strategy):
    def run(self, data: RecordsBucket, timestep: datetime):
        data = data.main
        if timestep == datetime.fromisoformat("2024-01-26 00:00:00"):
            self.broker.buy_long("AAPL", 100, 100, price_limit=(150, None))

def param2Dict(**kwargs):
    return kwargs
class TestJob(TestCase):
    def test_prep_brokers_data(self):
        # 1 create data pipe
        pipe = LoadPipe() | ToTSData()
        data = pipe.get(None, None)
        # Make data aapl marginable to True for the last day
        data[0]["AAPL"].data["Marginable"] = False
        data[0]["AAPL"].data["Marginable"].iloc[-1] = True
        data[0]["NVDA"].data["Marginable"] = False
        data[0]["NVDA"].data["Marginable"].iloc[-2:] = True
        # 2 create job
        params = param2Dict(metadata=Metadata(author="Tester", version="0.1.0"),
                 market_index=None,
                 main_timestep=0,
                 initial_cash=100_000,
                 commission=10,
                 relative_commission=None, margin_interest=0,
                 min_initial_margin=50, min_maintenance_margin=25,
                 liquidation_delay=2, min_initial_margin_short=50,
                 min_maintenance_margin_short=25,
                 window=50, default_marginable=True,
                 default_shortable=False,
                 risk_free_rate=1.5,
                 default_short_rate=1.5,
                 sell_at_the_end=True,
                 verbose=3)
        job = Job(MyStrat(), pipe, timedelta(days=90), params=params)
        job._data = data
        job._initialize_bcktst()
        prepared_data = job._prep_data(datetime(2024, 2, 21))

        # We take the main timestep: 0 (Which is the only one in this case
        yesterday_data, current_tick_data, marginables, dividends, div_freq, short_rate, security_names = job._prep_brokers_data(prepared_data[0])
        aapl = data[0]["AAPL"].data
        nvda = data[0]["NVDA"].data

        # 3 check if the data is correct
        np.testing.assert_allclose(aapl[["Open", "High", "Low", "Close"]].iloc[-2].values, yesterday_data[0])
        np.testing.assert_allclose(aapl[["Open", "High", "Low", "Close"]].iloc[-1].values, current_tick_data[0])
        np.testing.assert_allclose(nvda[["Open", "High", "Low", "Close"]].iloc[-2].values, yesterday_data[1])
        np.testing.assert_allclose(nvda[["Open", "High", "Low", "Close"]].iloc[-1].values, current_tick_data[1])
        np.testing.assert_array_equal([[False, False], [True, False]], marginables)
    def test_pipeline(self):
        """
        Test the pipeline in a simulated environment
        """
        # Days on which we are going to do the test
        days = ['2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25',
               '2024-01-26', '2024-01-29', '2024-01-30', '2024-01-31',
               '2024-02-01', '2024-02-02', '2024-02-05', '2024-02-06',
               '2024-02-07', '2024-02-08', '2024-02-09', '2024-02-12',
               '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16',
               '2024-02-20', '2024-02-21', '2024-02-22']
        days = pd.to_datetime(days)
        @Fetch
        def SimulatedFetch(frm: datetime, to: datetime, *args, **kwargs):
            chart1 = pd.read_csv("../engine_tests/test_data/AAPL_6mo_1d.csv", index_col="Date")
            chart2 = pd.read_csv("../engine_tests/test_data/NVDA_6mo_1d.csv", index_col="Date")
            chart1.index = pd.DatetimeIndex(["-".join(str(x).split("-")[:-1]) for x in chart1.index])
            chart2.index = pd.DatetimeIndex(["-".join(str(x).split("-")[:-1]) for x in chart2.index])
            chart1 = chart1.loc[frm:to]
            chart2 = chart2.loc[frm:to]
            return {"AAPL": chart1, "NVDA": chart2}

        @Fetch
        def IndexPipe(frm: datetime, to: datetime, *args, **kwargs):
            chart1 = pd.read_csv("../engine_tests/test_data/SPY_6mo_1d.csv", index_col="Date")
            chart1.index = pd.DatetimeIndex(["-".join(str(x).split("-")[:-1]) for x in chart1.index])
            chart1 = chart1.loc[frm:to]
            return {"SPY": chart1}

        pipe = SimulatedFetch() | ToTSData()
        index_pipe = IndexPipe() | ToTSData()
        params = param2Dict(metadata=Metadata(author="Tester", version="0.1.0"),
                 market_index=None,
                 main_timestep=0,
                 initial_cash=100_000,
                 commission=10,
                 relative_commission=None, margin_interest=0,
                 min_initial_margin=50, min_maintenance_margin=25,
                 liquidation_delay=2, min_initial_margin_short=50,
                 min_maintenance_margin_short=25,
                 window=50, default_marginable=True,
                 default_shortable=False,
                 risk_free_rate=1.5,
                 default_short_rate=1.5,
                 sell_at_the_end=True,
                 verbose=3)
        job = Job(MyStrat(), pipe, timedelta(days=90), params=params, index_pipe=index_pipe,
                  working_directory=PurePath("./.cache"))

        if os.path.exists(".cache"):
            shutil.rmtree(".cache")
        for day in days:
            job.pipeline(day)
"""
Note: This file doesn't run any unit test.  It is used to evaluate visually the reports generated.
"""
import pandas as pd
from datetime import datetime, timedelta
from src.backtest.engine import SimpleCashController, Metadata, RecordsBucket, Strategy, TSData
from src.backtest.data import Fetch, ToTSData, Process, PipeOutput
from src.backtest.serve.job import Job
from src.backtest.serve.renderers.email_renderer import EmailRenderer
from src.backtest.serve.renderers.html_renderer import HTMLRenderer
from src.backtest.serve.renderers.pdf_renderer import PDFRenderer
from pathlib import PurePath
import os
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MyStrat(Strategy):
    def run(self, data: RecordsBucket, timestep: datetime):
        # NVDA
        if timestep == datetime(2024, 1, 26):
            self.broker.buy_long("NVDA", 100)  # Buy all on the 29th of January at $612.32
        if timestep == datetime(2024, 2, 6):
            self.broker.sell_long("NVDA", 100, price_limit=(None, 1000), expiry=datetime(2024, 6, 30))

        # AAPL
        if timestep == datetime(2024, 1, 23):
            self.broker.sell_short("AAPL", 100)  # Will sell on the 24th of January at $195.1709
        # if timestep == datetime(2024, 2, 1):
        #     self.broker.buy_short("AAPL", 100)  # Will buy on the 2nd of February at $179.6308

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

@Process
def Flatten(frm: datetime, to: datetime, *args, po: PipeOutput[list[dict[str, TSData]]], **kwargs):
    return po.value[0]


renderer = EmailRenderer(style="light")

pipe = SimulatedFetch() | ToTSData()
index_pipe = IndexPipe() | ToTSData() | Flatten()
params = dict(metadata=Metadata(author="Tester", version="0.1.0"),
                    market_index=None,
                    main_timestep=0,
                    initial_cash=100_000,
                    commission=10,
                    relative_commission=None, margin_interest=0,
                    min_initial_margin=50, min_maintenance_margin=25,
                    liquidation_delay=2, min_initial_margin_short=50,
                    min_maintenance_margin_short=25,
                    window=50, default_marginable=True,
                    default_shortable=True,
                    risk_free_rate=1.5,
                    default_short_rate=1.5,
                    sell_at_the_end=True,
                    verbose=3)
job = Job(MyStrat(), pipe, timedelta(days=90), params=params, index_pipe=index_pipe,
          working_directory=PurePath("./.cache"), renderer=renderer,
          cash_controller=SimpleCashController(every_week=10.))

if os.path.exists(".cache"):
    shutil.rmtree(".cache")
for day in days:
    job.pipeline(day)
    if day == datetime(2024, 2, 29):
        break
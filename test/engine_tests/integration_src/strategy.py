from backtest import BackTest, Strategy, Metadata, TSData, DividendFrequency, Record, Records, RecordsBucket
from backtest.engine import CashControllerBase, BasicExtender
from datetime import datetime, timedelta
from typing import List, Tuple


class ComplexGoodStrategy(Strategy):
    """
    Hard coded strategy for testing purposes.
    Buy NVDA long with margin and AAPL short.
    This won't try any margin call nor bankruptcy.
    """
    def __init__(self):
        super().__init__()
    def run(self, data: RecordsBucket, timestep: datetime):

        # NVDA
        if timestep == datetime(2023, 11, 2):
            self.broker.buy_long("NVDA", 100, 100)    # Will buy on the 3rd of November at $440.1613
        if timestep == datetime(2023, 12, 13):
            self.broker.sell_long("NVDA", 100)    # Will sell on the 14th of December at $483.90
        if timestep == datetime(2024, 1, 4):
            self.broker.buy_long("NVDA", 75, 25)    # Will buy on the 5th of January at $484.62
        if timestep == datetime(2024, 2, 6):
            self.broker.sell_long("NVDA", 200)    # Sell all on the 7th of February at $683.19

        # AAPL
        if timestep == datetime(2023, 12, 12):
            self.broker.sell_short("AAPL", 200)    # Will sell on the 13th of December at $194.8414
        if timestep == datetime(2024, 1, 4):
            self.broker.buy_short("AAPL", 100)    # Will buy on the 5th of January at $181.758
        if timestep == datetime(2024, 1, 23):
            self.broker.sell_short("AAPL", 100)    # Will sell on the 24th of January at $195.1709
        if timestep == datetime(2024, 2, 1):
            self.broker.buy_short("AAPL", 200)    # Will buy on the 2nd of February at $179.6308

class WeekCashController(CashControllerBase):
    def every_week(self, timestamp: datetime) -> Tuple[float, str]:
        return 100, "Weekly deposit"


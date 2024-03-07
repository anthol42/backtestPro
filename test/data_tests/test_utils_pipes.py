from unittest import TestCase
from src.backtest.data.utils import JSONCache, JSONCacheObject
from src.backtest.data.pipes import Fetch, PipeOutput
from datetime import datetime, timedelta


class TestJSONCache(TestCase):
    def test_cache(self):
        expected = {
            "AAPL": [100, 101, 102, 103, 104],
            "MSFT": [200, 201, 202, 203, 204],
            "TSLA": [300, 301, 302, 303, 304]
        }
        @Fetch
        def FetchData(frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> dict:
            return expected

        # Build pipe
        pipe = FetchData | JSONCache(timeout=timedelta(seconds=5))
        # First run doesn't fetch cache
        out = pipe.get(datetime(2021, 1, 1), datetime(2021, 1, 5))
        self.assertEqual(expected, out)
        # Second run fetches cache
        out = pipe.get(datetime(2021, 1, 1), datetime(2021, 1, 5))
        self.assertEqual(expected, out)

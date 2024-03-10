from unittest import TestCase
from src.backtest.data.pipes import Fetch, Process, Collate, Cache, PipeOutput, RevalidateAction, CacheObject
from datetime import datetime, timedelta
import time
from typing import Optional
import json
import os

class TestPipes(TestCase):
    def test_fetch(self):
        @Fetch
        def FirstFetch(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            return ["This", "Is", "My", "Fetch"]

        self.assertEqual("FirstFetch", FirstFetch.name)
        self.assertEqual(["This", "Is", "My", "Fetch"],
                         FirstFetch.fetch(datetime.now(), datetime.now(), po=None).value)
        @Fetch
        def AppendFetch(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            return po.value + ["Appended"]

        self.assertEqual("AppendFetch", AppendFetch.name)
        pipe = FirstFetch | AppendFetch
        self.assertEqual(["This", "Is", "My", "Fetch", "Appended"], pipe.get(datetime.now(), datetime.now()))

    def test_process(self):
        @Process
        def MyProcess(frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs):
            return [value * 4 for value in po.value]

        self.assertEqual("MyProcess", MyProcess.name)
        self.assertEqual([4, 8, 12, 16, 20],
                         MyProcess.process(datetime.now(), datetime.now(), po=PipeOutput([1, 2, 3, 4, 5], None)).value)
        @Process
        def Divide(frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs):
            return [value / 2 for value in po.value]

        @Fetch
        def MakeN(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            return list(range(1, 11))

        pipe = MakeN | MyProcess | Divide
        self.assertEqual([2., 4., 6., 8., 10., 12., 14., 16., 18., 20.], pipe.get(datetime.now(), datetime.now()))

    def test_collate(self):

        @Fetch
        def FetchA(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            return list(range(1, 6))

        @Fetch
        def FetchB(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            return list(range(6, 11))

        @Collate
        def Concat(frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs):
            return po1.value + po2.value

        self.assertEqual("Concat", Concat.name)
        pipe = Concat(FetchA, FetchB)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pipe.get(datetime.now(), datetime.now()))

    def test_cache(self):
        if os.path.exists(f".cache/{1}.pkl"):
            os.remove(f".cache/{1}.pkl")
        if os.path.exists(f".cache/{1}.json"):
            os.remove(f".cache/{1}.json")
        i = 0
        @Fetch
        def FetchN(frm: datetime, to: datetime, *args, po: Optional[PipeOutput], **kwargs):
            out = list(range(i, i + 5))
            return out

        # Try to make a custom method to store and load cache from disk
        # We will save the object as JSON.
        # We will try to revalidate the cache every 1 second
        def JSON_load(pipe_id: int) -> Optional[CacheObject]:
            if os.path.exists(f".cache/{pipe_id}.json"):
                with open(f".cache/{pipe_id}.json", "r") as f:
                    data = json.load(f)
                    return CacheObject(data["data"], pipe_id, datetime.fromisoformat(data["revalidate"]),
                                       data["max_requests"], data["current_n_requests"])
            else:
                return None
        @Cache(loading_cb=JSON_load, store=True, timeout=timedelta(seconds=1))
        def MyCache(frm: datetime, to: datetime, *args, po: PipeOutput, pipe_id: int, revalidate: datetime,
                             timeout: timedelta, max_requests: int, n_requests: int, **kwargs):
            value = {
                "data": po.value,
                "stored_dt": datetime.now().isoformat(),
                "revalidate": revalidate.isoformat(),
                "current_n_requests": n_requests,
                "timeout": timeout.total_seconds(),
                "max_requests": max_requests
            }
            if not os.path.exists(".cache"):
                os.mkdir(".cache")
            with open(f".cache/{pipe_id}.json", "w") as f:
                json.dump(value, f)


        self.assertEqual("MyCache", MyCache.name)
        pipe = FetchN | MyCache
        out = pipe.get(datetime.now(), datetime.now())
        # Change the value returned by FetchN
        i += 1
        self.assertEqual([0, 1, 2, 3, 4], out)
        # Try to get the cache from disk
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([0, 1, 2, 3, 4], out)
        time.sleep(2)
        # Now, revalidate cache
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([1, 2, 3, 4, 5], out)

        # ----------------------------------------------
        # New pipe using the default caching method
        # ----------------------------------------------
        pipe = FetchN | Cache(timeout=timedelta(seconds=1))
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([1, 2, 3, 4, 5], out)
        # Change the value returned by FetchN
        i += 1
        # Try to load the cache from disk
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([1, 2, 3, 4, 5], out)

        # Revalidate cache
        time.sleep(2)
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([2, 3, 4, 5, 6], out)

        # ----------------------------------------------
        # New pipe using a custom revalidating method
        # ----------------------------------------------
        j: bool = False
        def revalidate(frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
            if not j:
                return RevalidateAction.NOOP
            else:
                return RevalidateAction.REVALIDATE

        # Reset cache to avoid bugs
        if os.path.exists(f".cache/{1}.pkl"):
            os.remove(f".cache/{1}.pkl")
        pipe = FetchN | Cache(revalidate_cb=revalidate, timeout=timedelta(seconds=1))
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([2, 3, 4, 5, 6], out)
        # Change the value returned by FetchN
        i += 1
        # Try to load the cache from disk
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([2, 3, 4, 5, 6], out)

        # Revalidate cache
        j = True
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([3, 4, 5, 6, 7], out)

        # ----------------------------------------------
        # New pipe trying to revalidate using the max requests
        # ----------------------------------------------
        # Reset cache to avoid bugs
        if os.path.exists(f".cache/{1}.pkl"):
            os.remove(f".cache/{1}.pkl")
        pipe = FetchN | Cache(max_requests=2)
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([3, 4, 5, 6, 7], out)
        # Change the value returned by FetchN
        i += 1
        # Try to load the cache from disk
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([3, 4, 5, 6, 7], out)
        # Change the value returned by FetchN
        i += 1
        # Try to load the cache from disk
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([3, 4, 5, 6, 7], out)
        out = pipe.get(datetime.now(), datetime.now())
        self.assertEqual([5, 6, 7, 8, 9], out)






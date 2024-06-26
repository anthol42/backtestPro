from typing import Optional
from unittest import TestCase
from datetime import datetime
from src.backtest.data.datapipe import DataPipe, DataPipeType, PipeOutput, CacheObject, RevalidateAction
from src.backtest.data.pipes import Fetch, Process, Collate, Cache
from src.backtest.data.utils import JSONCache

class TestDatapipe(TestCase):
    def test_piping(self):
        """
        Test the piping mechanism
        """
        # This will record the order of each pipe call
        call_order = []
        class Pipe(DataPipe):
            def __init__(self, T: DataPipeType, name: str = "", rev: bool = False):
                super().__init__(T)
                self.name = name
                self.rev = rev

            def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([1, 1, 1, 1, 1], self)

            def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([v * 2 for v in po.value], self)

            def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput,
                        **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([v1 + v2 for v1, v2 in zip(po1.value, po2.value)], self)
            def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
                call_order.append(self.name)
                self._cache = CacheObject(po, self._pipe_id, self.hash())

            def load(self) -> Optional[CacheObject]:
                return self._cache

            def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
                if self.rev:
                    call_order.append(self.name)
                    return RevalidateAction.FULL_REVALIDATE
                else:
                    call_order.append(self.name)
                    return RevalidateAction.NOOP

        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1", rev=True)
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2")
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
        result = pipe4.get(datetime.now(), datetime.now())
        expected_result = [8, 8, 8, 8, 8]
        expected_order = ["Fetch1", "Process1", "Cache1", "Fetch2", "process2", "Cache2", "Collate1", "Process3"]
        self.assertEqual(expected_result, result)
        self.assertEqual(expected_order, call_order)

    def test_caching(self):
        """
        Test the default caching mechanism.
        """
        call_order = []
        class Pipe(DataPipe):
            def __init__(self, T: DataPipeType, name: str = "", rev: bool = False):
                super().__init__(T)
                self.name = name
                self.rev = rev

            def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([1, 1, 1, 1, 1], self)

            def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([v * 2 for v in po.value], self)

            def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput,
                        **kwargs) -> PipeOutput:
                call_order.append(self.name)
                return PipeOutput([v1 + v2 for v1, v2 in zip(po1.value, po2.value)], self)
            def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
                call_order.append(self.name)
                self._cache = CacheObject(po.value, self._pipe_id, self.hash())

            def load(self) -> Optional[CacheObject]:
                return self._cache

            def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
                if self.rev:
                    call_order.append(self.name)
                    return RevalidateAction.FULL_REVALIDATE
                else:
                    call_order.append(self.name)
                    return RevalidateAction.NOOP
        # Building the pipe
        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1", rev=False)
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2", rev=False)
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")

        # Now, we are going to call the pipe twice, the second time, the cache should be used and there should be no
        # full revalidation.
        _ = pipe4.get(datetime.now(), datetime.now())
        results = pipe4.get(datetime.now(), datetime.now())
        expected_results = [8, 8, 8, 8, 8]
        expected_order = ["Fetch1", "Process1", "Cache1", "Fetch2", "process2", "Cache2", "Collate1", "Process3",
                          "Cache1", "Cache2", "Collate1", "Process3"]
        self.assertEqual(expected_results, results)
        self.assertEqual(expected_order, call_order)

        # Now, build a new pipe that will trigger full revalidation for the top branch
        call_order = []
        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1", rev=False)
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2", rev=True)
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
        _ = pipe4.get(datetime.now(), datetime.now())
        results = pipe4.get(datetime.now(), datetime.now())
        expected_order = ["Fetch1", "Process1", "Cache1", "Fetch2", "process2", "Cache2", "Collate1", "Process3",
                          "Cache1", "Cache2", "Fetch1", "Process1", "Cache1", "Fetch2", "process2", "Cache2", "Collate1", "Process3"]
        self.assertEqual(expected_order, call_order)
        self.assertEqual(expected_results, results)

        # Now, test a full revalidation with a pipe with multiple branches
        call_order = []
        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1")
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2")
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
        branch1 = Pipe(DataPipeType.FETCH, "Fetch4") | Pipe(DataPipeType.PROCESS, "Process4") | Pipe(DataPipeType.CACHE,
                                                                                                     "Cache4", rev=True)
        pipe = Pipe.Collate(pipe4, branch1) | Pipe(DataPipeType.PROCESS, "final_process")
        _ = pipe.get(datetime.now(), datetime.now())
        results = pipe.get(datetime.now(), datetime.now())
        first_run_order = ["Fetch1", "Process1", "Cache1", "Fetch2", "process2", "Cache2", "Collate1", "Process3",
                          "Fetch4", "Process4", "Cache4", "Collate", "final_process"]
        expected_order = first_run_order + ["Cache1", "Cache2", "Collate1", "Process3", "Cache4"] + first_run_order
        expected_results = [20, 20, 20, 20, 20]
        self.assertEqual(expected_order, call_order)
        self.assertEqual(expected_results, results)

    def test_id(self):
        """
        Test whether each given ID are unique
        """
        class Pipe(DataPipe):
            def __init__(self, T: DataPipeType, name: str = "", rev: bool = False):
                super().__init__(T)
                self.name = name
                self.rev = rev

            def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                return PipeOutput([1, 1, 1, 1, 1], self)

            def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                return PipeOutput([v * 2 for v in po.value], self)

            def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput,
                        **kwargs) -> PipeOutput:
                return PipeOutput([v1 + v2 for v1, v2 in zip(po1.value, po2.value)], self)
            def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
                self._cache = CacheObject(po, self._pipe_id)

            def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
                if self.rev:
                    return RevalidateAction.FULL_REVALIDATE
                else:
                    return RevalidateAction.NOOP

        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1")
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2")
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
        branch1 = Pipe(DataPipeType.FETCH, "Fetch4") | Pipe(DataPipeType.PROCESS, "Process4") | Pipe(DataPipeType.CACHE,
                                                                                                     "Cache4", rev=True)
        pipe = Pipe.Collate(pipe4, branch1) | Pipe(DataPipeType.PROCESS, "final_process")

        ids = set()
        for p in pipe:
            if p.pipe_id in ids:
                raise ValueError(f"Pipe ID {p.pipe_id} is not unique!")
            else:
                ids.add(p.pipe_id)

        self.assertEqual(13, len(ids))

    def test_ids(self):
        """
        Multiple pipes
        """

        @Fetch
        def LoadInt(frm: datetime, to: datetime, *args, **kwargs) -> list[int]:
            return [1, 2, 3, 4, 5]

        @Process
        def AddOne(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x + 1 for x in po.value]

        @Process
        def Double(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x * 2 for x in po.value]

        @Collate
        def Sum(frm: datetime, to: datetime, *args, po1: PipeOutput[list[int]], po2: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [v1 + v2 for v1, v2 in zip(po1.value, po2.value)]
        DataPipe.LAST_ID = 0   # Reset the ID counter
        branchA = LoadInt() | AddOne()
        branchB = LoadInt() | Double()
        pipe1 = Sum(branchA, branchB)
        _ = pipe1.get(datetime.now(), datetime.now())
        branchC = LoadInt() | AddOne()
        branchD = LoadInt() | Double()
        pipe2 = Sum(branchC, branchD)
        _ = pipe2.get(datetime.now(), datetime.now())
        pipe3 = Sum(branchC, branchB)
        _ = pipe3.get(datetime.now(), datetime.now())
        self.assertEqual([0, 1, 2, 3, 4], [p.pipe_id for p in pipe1])
        self.assertEqual([5, 6, 7, 8, 9], [p.pipe_id for p in pipe2])
        self.assertEqual([5, 6, 2, 3, 10], [p.pipe_id for p in pipe3])

    def test_set_id(self):
        """
        Multiple pipes
        """

        @Fetch
        def LoadInt(frm: datetime, to: datetime, *args, **kwargs) -> list[int]:
            return [1, 2, 3, 4, 5]

        @Process
        def AddOne(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x + 1 for x in po.value]

        @Process
        def Double(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x * 2 for x in po.value]

        @Collate
        def Sum(frm: datetime, to: datetime, *args, po1: PipeOutput[list[int]], po2: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [v1 + v2 for v1, v2 in zip(po1.value, po2.value)]
        DataPipe.LAST_ID = 0   # Reset the ID counter
        branchA = LoadInt() | AddOne()
        branchB = LoadInt() | Double()
        pipe1 = Sum(branchA, branchB)
        _ = pipe1.get(datetime.now(), datetime.now())
        branchC = LoadInt() | AddOne()
        branchD = LoadInt() | Double()
        pipe2 = Sum(branchC, branchD)
        pipe2.set_id(0)
        _ = pipe2.get(datetime.now(), datetime.now())
        self.assertEqual([0, 1, 2, 3, 4], [p.pipe_id for p in pipe1])
        self.assertEqual([0, 1, 2, 3, 4], [p.pipe_id for p in pipe2])

    def test_hash(self):
        """
        Test top-level hash with caching
        """
        data = [1, 2, 3, 4, 5]

        @Fetch
        def LoadInt(frm: datetime, to: datetime, *args, **kwargs) -> list[int]:
            return data

        @Process
        def AddOne(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x + 1 for x in po.value]

        @Process
        def Double(frm: datetime, to: datetime, *args, po: PipeOutput[list[int]], **kwargs) -> list[int]:
            return [x * 2 for x in po.value]

        @Collate
        def Sum(frm: datetime, to: datetime, *args, po1: PipeOutput[list[int]], po2: PipeOutput[list[int]], **kwargs) -> \
        list[int]:
            return [v1 + v2 for v1, v2 in zip(po1.value, po2.value)]

        branchA = LoadInt() | AddOne() | Cache()
        branchB = LoadInt() | Double() | JSONCache()
        pipe1 = Sum(branchA, branchB)
        res1 = pipe1.get(datetime.now(), datetime.now())
        data = [2, 2, 3, 4, 5]
        res2 = pipe1.get(datetime.now(), datetime.now())
        DataPipe.LAST_ID = 0  # Simulate a new run
        branchA = LoadInt() | Double() | Cache()
        branchB = LoadInt() | Double() | JSONCache()
        pipe1 = Sum(branchA, branchB)
        res3 = pipe1.get(datetime.now(), datetime.now())
        data = [1, 2, 3, 4, 5]
        res4 = pipe1.get(datetime.now(), datetime.now())
        self.assertEqual([4, 7, 10, 13, 16], res1)
        self.assertEqual([4, 7, 10, 13, 16], res2)
        self.assertEqual([8, 8, 12, 16, 20], res3)
        self.assertEqual([8, 8, 12, 16, 20], res4)

    def test_len(self):
        class Pipe(DataPipe):
            def __init__(self, T: DataPipeType, name: str = "", rev: bool = False):
                super().__init__(T)
                self.name = name
                self.rev = rev

            def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                return PipeOutput([1, 1, 1, 1, 1], self)

            def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
                return PipeOutput([v * 2 for v in po.value], self)

            def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput,
                        **kwargs) -> PipeOutput:
                return PipeOutput([v1 + v2 for v1, v2 in zip(po1.value, po2.value)], self)

            def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
                self._cache = CacheObject(po, self._pipe_id)

            def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
                if self.rev:
                    return RevalidateAction.FULL_REVALIDATE
                else:
                    return RevalidateAction.NOOP
        pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache1")
        pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE,
                                                                                                   "Cache2")
        pipe3 = Pipe.Collate(pipe1, pipe2)
        pipe3.name = "Collate1"
        pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
        branch1 = Pipe(DataPipeType.FETCH, "Fetch4") | Pipe(DataPipeType.PROCESS, "Process4") | Pipe(DataPipeType.CACHE,
                                                                                                     "Cache4", rev=True)
        pipe = Pipe.Collate(pipe4, branch1) | Pipe(DataPipeType.PROCESS, "final_process")
        self.assertEqual(13, len(pipe))
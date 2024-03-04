from typing import List, Tuple, Union, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from copy import deepcopy
from datapipe import DataPipe, RevalidateAction, DataPipeType, PipeOutput, CacheObject


class Fetch(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., Optional[PipeOutput], ...], PipeOutput]):
        super().__init__(DataPipeType.FETCH, name=cb.__name__)
        self._cb = cb

    def fetch(self, start: datetime, end: datetime, *args, po: PipeOutput,  **kwargs) -> PipeOutput:
        return self._cb(start, end, *args, **kwargs)

class Process(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput, ...], PipeOutput]):
        super().__init__(DataPipeType.PROCESS, name=cb.__name__)
        self._cb = cb

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        return self._cb(frm, to, *args, po, **kwargs)


class Collate(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput, PipeOutput, ...], PipeOutput]):
        super().__init__(DataPipeType.COLLATE, name=cb.__name__)
        self._cb = cb

    def __call__(self, *args: List[DataPipe]) -> DataPipe:
        self._pipes = args
        return self

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
        return self._cb(frm, to, *args, po1, po2, **kwargs)

class Cache(DataPipe):
    def __init__(self,
                 revalidate_cb: Callable[[datetime, datetime, ..., PipeOutput, ...], PipeOutput] = None, *,
                 revalidate: Optional[datetime] = None,
                 timeout: Optional[timedelta] = None,
                 max_request: Optional[int] = None,
                 store: bool = True):
        super().__init__(DataPipeType.CACHE)
        if revalidate_cb is not None:
            self._cb = revalidate_cb
            self.name = revalidate_cb.__name__
        else:
            self._cb = None
            self.name = "Cache"

        self._revalidate = revalidate
        self._timeout = timeout
        if timeout is not None:
            with_timeout = datetime.now() + timeout
            self._revalidate = min(self._revalidate, with_timeout) if self._revalidate is not None else with_timeout
        self._max_request = max_request
        self.store = store
        self._n_requests = 0

    def __call__(self, revalidate_cb: Callable[[datetime, datetime, ..., PipeOutput, ...], PipeOutput] = None):
        if revalidate_cb is not None:
            self._cb = revalidate_cb
            self.name = revalidate_cb.__name__
        else:
            self._cb = None
            self.name = "Cache"
        return self

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
        self._cache = CacheObject(po, self._pipe_id, self._revalidate, self._max_request)
        if self.store:
            self._cache.store()

    def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
        if self._cb is not None:
            return self._cb(frm, to, *args, po, **kwargs)
        else:    # Default implementation of the revalidate action
            if self._n_requests >= self._max_request:
                self._n_requests = 0
                return RevalidateAction.REVALIDATE
            elif self._revalidate is not None and datetime.now() >= self._revalidate:
                self._n_requests = 0
                return RevalidateAction.REVALIDATE
            else:
                self._n_requests += 1
                return RevalidateAction.NOOP



if __name__ == "__main__":
    @Cache(revalidate=timedelta(days=1))
    def MyCache(frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
        return po1.value + po2.value

    print(MyCache)

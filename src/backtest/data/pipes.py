from typing import List, Tuple, Union, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import os
from .datapipe import DataPipe, RevalidateAction, DataPipeType, PipeOutput, CacheObject


class Fetch(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., Optional[PipeOutput[Any]], ...], Any]):
        super().__init__(DataPipeType.FETCH, name=cb.__name__)
        self._cb = cb

    def fetch(self, start: datetime, end: datetime, *args, po: Optional[PipeOutput],  **kwargs) -> PipeOutput:
        return PipeOutput(self._cb(start, end, *args, po=po, **kwargs), self)

class Process(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput[Any], ...], Any]):
        super().__init__(DataPipeType.PROCESS, name=cb.__name__)
        self._cb = cb

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        return PipeOutput(self._cb(frm, to, *args, po=po, **kwargs), self)


class Collate(DataPipe):
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput[Any], PipeOutput, ...], Any]):
        super().__init__(DataPipeType.COLLATE, name=cb.__name__)
        self._cb = cb

    def __call__(self, po1: DataPipe, po2: DataPipe) -> DataPipe:
        self._pipes = [po1, po2]
        return self

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput[Any], po2: PipeOutput, **kwargs) -> PipeOutput:
        return PipeOutput(self._cb(frm, to, *args, po1=po1, po2=po2, **kwargs), self)

class Cache(DataPipe):
    def __init__(self,
                 caching_cb: Callable[[datetime, datetime, ..., PipeOutput[Any], int, ...], None] = None, *,
                 loading_cb: Callable[[int], CacheObject] = None,
                 revalidate: Optional[datetime] = None,
                 timeout: Optional[timedelta] = None,
                 max_requests: Optional[int] = None,
                 store: bool = True,
                 revalidate_cb: Callable[[datetime, datetime, Tuple[Any, ...], PipeOutput[Any], dict[str, Any]],
                    RevalidateAction] = None):
        super().__init__(DataPipeType.CACHE)
        if caching_cb is not None:
            self._caching_cb = caching_cb
            self.name = caching_cb.__name__
        else:
            self._caching_cb = None
            self.name = "Cache"

        self._revalidate = revalidate
        self._revalidate_cb = revalidate_cb
        self._loading_cb = loading_cb
        self._timeout = timeout
        if timeout is not None:
            with_timeout = datetime.now() + timeout
            self._revalidate = min(self._revalidate, with_timeout) if self._revalidate is not None else with_timeout
        self._max_request = max_requests
        self.store = store
        self._n_requests = 0

    def __call__(self, caching_cb: Callable[[datetime, datetime, Tuple[Any, ...], PipeOutput[Any], dict[str, Any]], PipeOutput] = None) -> DataPipe:
        if caching_cb is not None:
            self._caching_cb = caching_cb
            self.name = caching_cb.__name__
        else:
            self._caching_cb = None
            self.name = "Cache"
        return self

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> None:
        if self._caching_cb is None:
            self._cache = CacheObject(po.value, self._pipe_id, self._revalidate, self._max_request, self._n_requests)
            if self.store:
                self._cache.store()
        else:
            self._caching_cb(frm, to, *args, po=po, pipe_id=self.pipe_id, revalidate=self._revalidate,
                             timeout=self._timeout, max_requests=self._max_request, n_requests=self._n_requests,
                             **kwargs)

    def load(self) -> Optional[CacheObject]:
        if self._loading_cb is not None:
            return self._loading_cb(self._pipe_id)
        else:
            if os.path.exists(f".cache/{self._pipe_id}.pkl"):
                return CacheObject.load(self._pipe_id)
            else:
                return None

    def _load_cache(self):
        if self.store:
            self._cache = self.load()
            if self._cache is not None:
                self._revalidate = self._cache.next_revalidate
                self._n_requests = self._cache.current_n_requests

    def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> RevalidateAction:
        if self._revalidate_cb is not None:
            return self._revalidate_cb(frm, to, *args, po=po, **kwargs)
        else:    # Default implementation of the revalidate action
            if self._max_request is not None and self._n_requests >= self._max_request:
                self._n_requests = 0
                if self._timeout is not None:
                    self._revalidate = datetime.now() + self._timeout
                return RevalidateAction.REVALIDATE
            elif self._revalidate is not None and datetime.now() >= self._revalidate:
                self._n_requests = 0
                if self._timeout is not None:
                    self._revalidate = datetime.now() + self._timeout
                return RevalidateAction.REVALIDATE
            else:
                self._n_requests += 1
                if self._max_request is not None:    # Means we are monitoring the number of requests
                    self.cache(frm, to, *args, po=PipeOutput(self._cache.value, self), **kwargs)    # Update the number of requests in the cache
                return RevalidateAction.NOOP



if __name__ == "__main__":
    @Cache(revalidate=timedelta(days=1))
    def MyCache(frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
        return po1.value + po2.value

    print(MyCache)

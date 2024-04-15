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
from typing import List, Tuple, Union, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import os
from .datapipe import DataPipe, RevalidateAction, DataPipeType, PipeOutput, CacheObject
from copy import deepcopy


class Fetch(DataPipe):
    """
    Function decorator designed to make a Fetch pipe out of a function.
    It is designed to fetch data from a source. The source can be a database, a web API, a file, etc.
    The pipe is named after the function name.
    """
    def __init__(self, cb: Callable[[datetime, datetime, ..., Optional[PipeOutput[Any]], ...], Any]):
        """
        :param cb: The fetch callback method.
        """
        super().__init__(DataPipeType.FETCH, name=cb.__name__)
        self._cb = cb
        self.__wrapped__ = self._cb
        self._called = False

    def __call__(self):
        self._called = True
        return deepcopy(self)

    def fetch(self, start: datetime, end: datetime, *args, po: Optional[PipeOutput],  **kwargs) -> PipeOutput:
        if not self._called:
            raise RuntimeError(f"You must call the object when building your pipe.  This pipe '{self.name}' wasn't called.\n"
                               f"Example: pipe = MyFetch().\n"
                               f"Not: pipe = MyFetch")
        return PipeOutput(self._cb(start, end, *args, po=po, **kwargs), self)

class Process(DataPipe):
    """
    Function decorator designed to make a Process pipe out of a function.
    It is designed to process data. The data can be transformed, cleaned, imputed, etc.
    The pipe is named after the function name.
    """
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput[Any], ...], Any]):
        """
        :param cb: The process callback method.
        """
        super().__init__(DataPipeType.PROCESS, name=cb.__name__)
        self._cb = cb
        self.__wrapped__ = self._cb
        self._called = False

    def __call__(self):
        self._called = True
        return deepcopy(self)

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        if not self._called:
            raise RuntimeError(f"You must call the object when building your pipe.  This pipe '{self.name}' wasn't called.\n"
                               f"Example: pipe = MyProcess().\n"
                               f"Not: pipe = MyProcess")
        return PipeOutput(self._cb(frm, to, *args, po=po, **kwargs), self)


class Collate(DataPipe):
    """
    Function decorator designed to make a Collate pipe out of a function.
    It is designed to collate data. The data can be combined, aggregated, concatenated, etc.
    The pipe is named after the function name.
    """
    def __init__(self, cb: Callable[[datetime, datetime, ..., PipeOutput[Any], PipeOutput, ...], Any]):
        """
        :param cb: The collate callback method.
        """
        super().__init__(DataPipeType.COLLATE, name=cb.__name__)
        self._cb = cb
        self.__wrapped__ = self._cb

    def __call__(self, po1: DataPipe, po2: DataPipe) -> DataPipe:
        new = deepcopy(self)
        new._pipes = [po1, po2]
        # Increment the pipe_id of the second branch and all its children
        new._pipe_id = po2._increment_id(po1._pipe_id + 1)
        return new

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput[Any], po2: PipeOutput, **kwargs) -> PipeOutput:
        return PipeOutput(self._cb(frm, to, *args, po1=po1, po2=po2, **kwargs), self)

class Cache(DataPipe):
    """
    Function decorator designed to make a Cache pipe out of a function.
    The decorated function is expected to be the caching method.  When using this, it is recommended to pass a load
    callback method to load the cache object from the storage.  The caching method should be able to store the cache
    and load it from the storage.  The cache object should be a CacheObject instance.  It is also possible to pass a
    revalidate callback method to define a dynamic revalidate action.  The revalidate action is a method that returns a
    RevalidateAction enum value.  The revalidate action is used to determine if the cache should be revalidated or not.
    If no revalidate method is passed, the default revalidate method check if the cache is expired given an expired
    datetime, a timeout or a maximum number of requests.

    Examples:
    >>> @Cache(loading_cb=JSON_load, store=True, timeout=timedelta(seconds=1))
    ... def MyCache(frm: datetime, to: datetime, *args, po: PipeOutput, pipe_id: int, revalidate: datetime,
    ...          timedelta, max_requests: int, n_requests: int, **kwargs):
    ...     value = {
    ...         "data": po.value,
    ...         "stored_dt": datetime.now().isoformat(),
    ...         "revalidate": revalidate.isoformat(),
    ...         "current_n_requests": n_requests,
    ...         "timeout": timeout.total_seconds(),
    ...         "max_requests": max_requests
    ...     }
    ...     if not os.path.exists(".cache"):
    ...         os.mkdir(".cache")
    ...     with open(f".cache/{pipe_id}.json", "w") as f:
    ...         json.dump(value, f)
    >>> # The cache pipe can also be used as is:
    >>> pipe = FetchN | Cache(timeout=timedelta(seconds=1))

    """
    def __init__(self,
                 caching_cb: Callable[[datetime, datetime, ..., PipeOutput[Any], int, DataPipe, ...], None] = None, *,
                 loading_cb: Callable[[int], CacheObject] = None,
                 revalidate: Optional[datetime] = None,
                 timeout: Optional[timedelta] = None,
                 max_requests: Optional[int] = None,
                 store: bool = True,
                 revalidate_cb: Callable[[datetime, datetime, Tuple[Any, ...], PipeOutput[Any], dict[str, Any]],
                    RevalidateAction] = None):
        """
        :param caching_cb: The callback that is called after the pipeline wrapped by the Cache pipe is executed.  It is used to store the cache.
        :param loading_cb: The callback used to load the cache from the disk.
        :param revalidate: The next datetime to revalidate the cache.
        :param timeout: A timedelta object representing the max age of the cache before revalidating it.  If it is one
        day, the cache will be revalidated every day as opposed to the revalidate parameter, which revalidate only once.
        :param max_requests: The maximum number of request that hit the cache before revalidating it.
        :param store: Whether to store the cache on disk or keep it only in memory.
        :param revalidate_cb: A callback method that returns a RevalidateAction enum value.  It is used to determine if
        the cache should be revalidated or not. This callback can be provided for complex revalidation logic.
        """
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
        new = deepcopy(self)
        if caching_cb is not None:
            new._caching_cb = caching_cb
            new.__wrapped__ = caching_cb
            new.name = caching_cb.__name__
        return new

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> None:
        if self._caching_cb is None:
            self._cache = CacheObject(po.value, self._pipe_id, self.hash(), self._revalidate, self._max_request,
                                      self._n_requests)
            if self.store:
                self._cache.store()
        else:
            self._caching_cb(frm, to, *args, po=po, pipe_id=self.pipe_id, self=self, revalidate=self._revalidate,
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
        if self._cache is not None and self._cache.pipe_hash != self.hash():
            print("Pipe has changed.  Performing full revalidation.")
            return RevalidateAction.FULL_REVALIDATE
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

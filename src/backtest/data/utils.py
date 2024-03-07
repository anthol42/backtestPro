from .pipes import Fetch, Process, Collate, Cache, PipeOutput, RevalidateAction, CacheObject, DataPipeType, DataPipe
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple, Any, Callable
from . import json_extension as je
import os

class JSONCacheObject(CacheObject):
    def store(self):
        with open(f".cache/{self.pipe_id}.json", "w") as file:
            out = {
                "value": self.value,
                "pipe_id": self.pipe_id,
                "write_time": self.write_time,
                "next_revalidate": self.next_revalidate,
                "max_request": self.max_request,
                "current_n_request": self.current_n_requests
            }
            json.dump(out, file, cls=je.JSONEncoder)

    @classmethod
    def load(cls, pipe_id: int) -> 'JSONCacheObject':
        with open(f".cache/{pipe_id}.json", "r") as file:
            data = json.load(file, cls=je.JSONDecoder)
            new_cache = cls(data["value"], data["pipe_id"], data["next_revalidate"],
                       data["max_request"], data["current_n_request"])
            new_cache.write_time = data["write_time"]
        return new_cache

class JSONCache(Cache):
    """
    This pipe will cache the data as a JSON file instead of a pickle file.  This class uses an extended JSON encoder
    and decoder to encode most datatypes.  However, the encoded version might be suboptimal.  To make objects
    JSON serializable, you can add a __tojson__ method that returns a JSONable dictionary to your object.  To make it
    loadable, you can add a __fromjson__ class method that takes a JSONable dictionary and returns an instance of your
    object.
    """
    def __init__(self, *,
                 revalidate: Optional[datetime] = None,
                 timeout: Optional[timedelta] = None,
                 max_requests: Optional[int] = None,
                 store: bool = True,
                 revalidate_cb: Callable[[datetime, datetime, Tuple[Any, ...], PipeOutput, dict[str, Any]],
                    RevalidateAction] = None,
                 custom_types: Optional[dict[str, type]] = None):
        """
        :param revalidate: The datetime to revalidate the cache
        :param timeout: The timedelta at which the cache will expire and be revalidated periodically
        :param max_requests: The maximum number of request before revaidating the cache
        :param store: Whether to store on disk or only in memory
        :param revalidate_cb: A revalidate callback function to implement a custom revalidation mechanism
        :param custom_types: The custom types to add to the JSONEncoder and JSONDecoder to make them JSON serializable
        and loadable.  Every type that implemented the __tojson__ and __fromjson__ methods should be registered here.
        <class_name: class>
        """
        super().__init__(revalidate=revalidate, timeout=timeout, max_requests=max_requests,
                         store=store, revalidate_cb=revalidate_cb)
        self._custom_types = custom_types if custom_types is not None else {}
        self.name = "JSONCache"

    def __call__(self, *args, **kwargs):
        """
        This object is not implemented for JSONCache, use Cache pipe instead.  This method is
        not meant to be used as a decorator nor to be called directly.
        :param args:
        :param kwargs:
        :raise NotImplementedError:
        """
        raise NotImplementedError("This method is not implemented for JSONCache, use Cache pipe instead")

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
        """
        This method is called to cache the data.  This emthod will cache the data to the json format, and save it
        to the disk.  It will also save as pickle the automatically detected types.
        :param frm: Start Datetime
        :param to: End Datetime
        :param args: Any arguments passed to the pipeline
        :param po: The previous PipeOutput
        :param kwargs: Any keyword arguments passed to the pipeline
        :return: None
        """
        self._cache = JSONCacheObject(po.value, self._pipe_id, self._revalidate, self._max_request, self._n_requests)
        if self.store:
            je.add_types(**self._custom_types)
            self._cache.store()
            je.remove_types(*self._custom_types.keys())

    def load(self) -> Optional[JSONCacheObject]:
        """
        This method is called to load the cache.  This method will load the cache from the disk, and return the
        CacheObject.  If the cache is not found, it will return None.
        :return: JSONCacheObject if cache is found and None otherwise
        """
        if os.path.exists(f".cache/{self._pipe_id}.json"):
            je.add_types(**self._custom_types)
            out = JSONCacheObject.load(self._pipe_id)
            je.remove_types(*self._custom_types.keys())
            return out
        else:
            return None

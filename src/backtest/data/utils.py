from .pipes import Fetch, Process, Collate, Cache, PipeOutput, RevalidateAction, CacheObject, DataPipeType, DataPipe
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple, Any, Callable
from .json_extension import JSONEncoder


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
                    RevalidateAction] = None):
        super().__init__(revalidate=revalidate, timeout=timeout, max_requests=max_requests,
                         store=store, revalidate_cb=revalidate_cb)

    def __call__(self, *args, **kwargs):
        """
        This object is not implemented for JSONCache, use Cache pipe instead.  This method is
        not meant to be used as a decorator nor to be called directly.
        :param args:
        :param kwargs:
        :raise NotImplementedError:
        """
        raise NotImplementedError("This method is not implemented for JSONCache, use Cache pipe instead")

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        pass

    def load(self) -> Optional[CacheObject]:
        pass

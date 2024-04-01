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
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time

from enum import Enum

TYPES: dict[str, type] = {}
DETECTED_TYPES: dict[str, type] = {}


def add_types(**types):
    """
    Add new types supported by extended json encoder/decoder.  The type will be serialized using the __tojson__ method.
    (Must exist in the object) and deserialized using the __fromjson__ class method.
    :param types: A dictionary of types to add to the JSONEncoder [str: class]
    """
    TYPES.update(types)

def remove_types(*types):
    """
    Remove supported types from the extended json encoder/decoder.  The type will be serialized using the __tojson__ method.
    (Must exist in the object) and deserialized using the __fromjson__ class method.
    :param types: A list of types to remove from the JSONEncoder [str]
    """
    for t in types:
        TYPES.pop(t, None)

def get_detected_types() -> dict[str, type]:
    """
    Get the types that have been automatically detected by the JSONEncoder.
    :return: A dictionary of detected types [str: class]
    """
    return DETECTED_TYPES.copy()

def add_detected_types(types: dict[str, type]):
    """
    Load the detected types into the JSONDecoder.  Can be useful to restor the state of the module
    :param types: A dictionary of types to add to the JSONDecoder [str: class]
    """
    DETECTED_TYPES.update(types)

def set_detected_types(types: dict[str, type]):
    """
    Load the detected types into the JSONDecoder.  Can be useful to restor the state of the module
    :param types: A dictionary of types to add to the JSONDecoder [str: class]
    """
    global DETECTED_TYPES
    DETECTED_TYPES = types

class JSONEncoder(json.JSONEncoder):
    """
    This class extends the JSONEncoder class from the json module to handle the serialization of a variety of objects.
    If your object isn't serialized properly, you can add a __json__ method to your object to handle the serialization
    and return a jsonable dictionary.
    """
    def default(self, o):
        return self._recursive_json(o)

    def _recursive_json(self, o):
        default_types = {int, float, str, list, tuple, dict, bool, type(None)}
        if type(o) in default_types:
            return o
        elif hasattr(o, "__tojson__"):
            if type(o).__name__ in TYPES:
                return {"__TYPE__": type(o).__name__, "data": o.__tojson__()}
            else:
                raise TypeError(f"Object of type {type(o)} is JSON serializable, but not registered in the JSONEncoder.  "
                                f"Use the add_types function to add the type to the JSONEncoder.")
        elif isinstance(o, pd.DataFrame):
            return {"__TYPE__": "pd.DataFrame", "data": o.to_dict(orient="list"), "index": o.index.to_list()}
        elif isinstance(o, np.ndarray):
            return {"__TYPE__": "np.ndarray", "data": o.tolist()}
        elif isinstance(o, np.int64):
            return int(o)
        elif isinstance(o, np.float64):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.datetime64):
            return {"__TYPE__": "np.datetime64", "data": o.astype(str)}
        elif isinstance(o, np.timedelta64):
            return {"__TYPE__": "np.timedelta64", "data": o.item().total_seconds()}
        elif isinstance(o, datetime):
            return {"__TYPE__": "datetime", "data": o.isoformat()}
        elif isinstance(o, timedelta):
            return {"__TYPE__": "timedelta", "data": o.total_seconds()}
        elif isinstance(o, date):
            return {"__TYPE__": "date", "data": o.isoformat()}
        elif isinstance(o, time):
            return {"__TYPE__": "time", "data": o.isoformat()}
        elif isinstance(o, pd.Series):
            return {"__TYPE__": "pd.Series", "data": o.to_dict()}
        elif isinstance(o, Enum):
            DETECTED_TYPES[type(o).__name__] = o.__class__
            return {"__TYPE__": "enum", "enum_name": o.__class__.__name__, "data": o.value}
        elif hasattr(o, "__dict__"):
            DETECTED_TYPES[type(o).__name__] = o.__class__
            return {"__TYPE__": o.__class__.__name__, "data": {k: self._recursive_json(v) for k, v in o.__dict__.items()}}
        elif hasattr(o, "__iter__"):
            DETECTED_TYPES[type(o).__name__] = o.__class__
            return {"__TYPE__": o.__class__.__name__, "data": [self._recursive_json(i) for i in o]}
        else:
            raise TypeError(f"Object of type {type(o)} is not JSON serializable.  To make it serializable, add a "
                            f"__json__ method to the object that returns a dictionary JSONable.")


class JSONDecoder(json.JSONDecoder):
    """
    This class extends the JSONDecoder class from the json module to handle the deserialization of a variety of objects.
    If your object isn't deserialized properly, you can add a __fromjson__ method to your object to handle the
    deserialization and return the object.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self._recursive_fromjson, *args, **kwargs)

    def _recursive_fromjson(self, d):
        # Dict is not there because it could be anything (Custom type)
        default_types = {int, float, str, list, tuple, bool, type(None)}
        if type(d) in default_types:
            return d
        if "__TYPE__" in d:
            # Built-in extended types
            if d["__TYPE__"] == "pd.DataFrame":
                return pd.DataFrame(d["data"], index=d["index"])
            elif d["__TYPE__"] == "np.ndarray":
                return np.array(d["data"])
            elif d["__TYPE__"] == "pd.Series":
                index = list(d["data"].keys())
                if index == [str(i) for i in range(len(d["data"]))]:
                    # We have a range index
                    return pd.Series(d["data"].values(), index=range(len(d["data"])))
                else:
                    return pd.Series(d["data"])
            elif d["__TYPE__"] == "np.datetime64":
                return np.datetime64(d["data"])
            elif d["__TYPE__"] == "np.timedelta64":
                td = timedelta(seconds=d["data"])
                return np.timedelta64(td)
            elif d["__TYPE__"] == "datetime":
                return datetime.fromisoformat(d["data"])
            elif d["__TYPE__"] == "timedelta":
                return timedelta(seconds=d["data"])
            elif d["__TYPE__"] == "date":
                return date.fromisoformat(d["data"])
            elif d["__TYPE__"] == "time":
                return time.fromisoformat(d["data"])
            elif d["__TYPE__"] == "enum":
                return DETECTED_TYPES[d["enum_name"]](d["data"])
            # Custom extended types
            elif d["__TYPE__"] in TYPES:
                return TYPES[d["__TYPE__"]].__fromjson__(d["data"])
            # Autodetected types
            elif d["__TYPE__"] in DETECTED_TYPES and type(d["data"]) == dict:
                o = DETECTED_TYPES[d["__TYPE__"]].__new__(DETECTED_TYPES[d["__TYPE__"]])
                o.__dict__.update(d["data"])
                return o
            elif d["__TYPE__"] in DETECTED_TYPES and type(d["data"]) == list:
                return DETECTED_TYPES[d["__TYPE__"]]([i for i in d["data"]])
            else:
                raise TypeError(f"Object of type {d['__TYPE__']} is JSON deserializable, but not registered in the "
                                f"JSONDecoder.  Use the add_types function to add the type to the JSONDecoder.")
        elif isinstance(d, dict):
            return {k: v for k, v in d.items()}
        elif isinstance(d, list):
            return [i for i in d]
        else:
            return d
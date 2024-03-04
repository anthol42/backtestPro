import json
import pandas as pd
import numpy as np

TYPES = {}


def add_types(**types):
    """
    Add a new type to the JSONEncoder.  The type will be serialized using the __json__ method if it exists.
    :param types: A dictionary of types to add to the JSONEncoder [str: class]
    """
    TYPES.update(types)

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
            return {"__TYPE__": "pd.DataFrame", "data": o.to_dict(orient="list")}
        elif isinstance(o, np.ndarray):
            return {"__TYPE__": "np.ndarray", "data": o.tolist()}
        elif isinstance(o, np.int64):
            return int(o)
        elif isinstance(o, np.float64):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.datetime64):
            return o.astype(str)
        elif isinstance(o, np.timedelta64):
            return o.astype(str)
        elif isinstance(o, pd.Series):
            return {"__TYPE__": "pd.Series", "data": o.to_dict()}
        elif hasattr(o, "__dict__"):
            return {k: self._recursive_json(v) for k, v in o.__dict__.items()}
        elif hasattr(o, "__iter__"):
            return [self._recursive_json(i) for i in o]
        else:
            raise TypeError(f"Object of type {type(o)} is not JSON serializable.  To make it serializable, add a "
                            f"__json__ method to the object that returns a dictionary JSONable.")
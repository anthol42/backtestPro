from unittest import TestCase
from src.backtest.data.json_extension import JSONEncoder
import src.backtest.data.json_extension as je
import numpy as np
import pandas as pd
import json

class TestJSONEncoder(TestCase):
    def test_type_conversion(self):
        # Default types
        float_ = 1.0
        int_ = 1
        str_ = "1"
        list_ = [1, 2, 3]
        tuple_ = (1, 2, 3)
        dict_ = {"a": 1, "b": 2}
        bool_ = True
        none_ = None

        # Complex types
        set_ = {1, 2, 3}
        frozenset_ = frozenset({1, 2, 3})
        np_int64 = np.int64(1)
        np_float64 = np.float64(1.0)
        np_bool = np.bool_(True)
        np_datetime64 = np.datetime64("2023-08-22")
        np_timedelta64 = np.timedelta64(1, "D")
        pd_series = pd.Series([1, 2, 3])
        pd_dataframe = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test default types
        out = json.dumps(float_, cls=JSONEncoder)
        self.assertEqual(json.dumps(float_), out)
        out = json.dumps(int_, cls=JSONEncoder)
        self.assertEqual(json.dumps(int_), out)
        out = json.dumps(str_, cls=JSONEncoder)
        self.assertEqual(json.dumps(str_), out)
        out = json.dumps(list_, cls=JSONEncoder)
        self.assertEqual(json.dumps(list_), out)
        out = json.dumps(tuple_, cls=JSONEncoder)
        self.assertEqual(json.dumps(tuple_), out)
        out = json.dumps(dict_, cls=JSONEncoder)
        self.assertEqual(json.dumps(dict_), out)
        out = json.dumps(bool_, cls=JSONEncoder)
        self.assertEqual(json.dumps(bool_), out)
        out = json.dumps(none_, cls=JSONEncoder)
        self.assertEqual(json.dumps(none_), out)

        # Test complex types
        out = json.dumps(set_, cls=JSONEncoder)
        self.assertEqual(f"[1, 2, 3]", out)
        out = json.dumps(frozenset_, cls=JSONEncoder)
        self.assertEqual(f"[1, 2, 3]", out)
        out = json.dumps(np_int64, cls=JSONEncoder)
        self.assertEqual("1", out)
        out = json.dumps(np_float64, cls=JSONEncoder)
        self.assertEqual("1.0", out)
        out = json.dumps(np_bool, cls=JSONEncoder)
        self.assertEqual("true", out)
        out = json.dumps(np_datetime64, cls=JSONEncoder)
        self.assertEqual(f"\"2023-08-22\"", out)
        out = json.dumps(np_timedelta64, cls=JSONEncoder)
        self.assertEqual(f"\"1 days\"", out)
        out = json.dumps(pd_series, cls=JSONEncoder)
        self.assertEqual("{\"__TYPE__\": \"pd.Series\", \"data\": {\"0\": 1, \"1\": 2, \"2\": 3}}", out)
        out = json.dumps(pd_dataframe, cls=JSONEncoder)
        self.assertEqual("{\"__TYPE__\": \"pd.DataFrame\", \"data\": {\"a\": [1, 2, 3], \"b\": [4, 5, 6]}}", out)
        out = json.dumps(pd_dataframe, cls=JSONEncoder)
        self.assertEqual("{\"__TYPE__\": \"pd.DataFrame\", \"data\": {\"a\": [1, 2, 3], \"b\": [4, 5, 6]}}", out)

    def test_complex_types(self):
        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3
                self.serie = pd.Series([1, 2, 3])

        my_class = MyClass()
        out = json.dumps(my_class, cls=JSONEncoder)
        self.assertEqual("{\"a\": 1, \"b\": 2, \"c\": 3, \"serie\": {\"__TYPE__\": \"pd.Series\", \"data\": {\"0\": 1, \"1\": 2, \"2\": 3}}}", out)

    def test_complex_call__json__(self):
        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3
                self.serie = pd.Series([1, 2, 3])

            def __tojson__(self):
                return {"Hello": "World", "a": self.a, "b": self.b, "c": self.c, "serie": self.serie.to_dict()}

        my_class = MyClass()
        je.add_types(MyClass=MyClass)
        out = json.dumps(my_class, cls=JSONEncoder)
        expected = "{\"__TYPE__\": \"MyClass\", \"data\": {\"Hello\": \"World\", \"a\": 1, \"b\": 2, \"c\": 3, \"serie\": {\"0\": 1, \"1\": 2, \"2\": 3}}}"
        self.assertEqual(expected, out)

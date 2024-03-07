from unittest import TestCase
from src.backtest.data.json_extension import JSONEncoder, JSONDecoder
import src.backtest.data.json_extension as je
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta, date, time

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
        dt = datetime(2023, 8, 22)
        td = timedelta(days=1, hours=12)
        date_ = date(2023, 8, 22)
        time_ = time(12, 0)

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
        self.assertEqual('{"__TYPE__": "set", "data": [1, 2, 3]}', out)
        out = json.dumps(frozenset_, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "frozenset", "data": [1, 2, 3]}', out)
        out = json.dumps(np_int64, cls=JSONEncoder)
        self.assertEqual("1", out)
        out = json.dumps(np_float64, cls=JSONEncoder)
        self.assertEqual("1.0", out)
        out = json.dumps(np_bool, cls=JSONEncoder)
        self.assertEqual("true", out)
        out = json.dumps(np_datetime64, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "np.datetime64", "data": "2023-08-22"}', out)
        out = json.dumps(np_timedelta64, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "np.timedelta64", "data": 86400.0}', out)
        out = json.dumps(pd_series, cls=JSONEncoder)
        self.assertEqual("{\"__TYPE__\": \"pd.Series\", \"data\": {\"0\": 1, \"1\": 2, \"2\": 3}}", out)
        out = json.dumps(pd_dataframe, cls=JSONEncoder)
        self.assertEqual("{\"__TYPE__\": \"pd.DataFrame\", \"data\": {\"a\": [1, 2, 3], \"b\": [4, 5, 6]}}", out)
        out = json.dumps(dt, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "datetime", "data": "2023-08-22T00:00:00"}', out)
        out = json.dumps(td, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "timedelta", "data": 129600.0}', out)
        out = json.dumps(date_, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "date", "data": "2023-08-22"}', out)
        out = json.dumps(time_, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "time", "data": "12:00:00"}', out)


    def test_complex_types(self):
        je.DETECTED_TYPES = {}
        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3
                self.serie = pd.Series([1, 2, 3])

        my_class = MyClass()
        out = json.dumps(my_class, cls=JSONEncoder)
        self.assertEqual('{"__TYPE__": "MyClass", "data": {\"a\": 1, \"b\": 2, \"c\": 3, \"serie\": {\"__TYPE__\": \"pd.Series\", \"data\": {\"0\": 1, \"1\": 2, \"2\": 3}}}}', out)
        self.assertEqual({"MyClass": MyClass}, je.get_detected_types())
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

class TestJSONDecoder(TestCase):
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
        dt = datetime(2023, 8, 22)
        td = timedelta(days=1, hours=12)
        date_ = date(2023, 8, 22)
        time_ = time(12, 0)

        # Test default types
        out = json.loads(json.dumps(float_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(float_, out)
        out = json.loads(json.dumps(int_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(int_, out)
        out = json.loads(json.dumps(str_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(str_, out)
        out = json.loads(json.dumps(list_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(list_, out)
        out = json.loads(json.dumps(tuple_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(list(tuple_), out)    # By default, tuples are serialized as lists
        out = json.loads(json.dumps(dict_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(dict_, out)
        out = json.loads(json.dumps(bool_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(bool_, out)
        out = json.loads(json.dumps(none_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(none_, out)

        # Test complex types
        # Need to set the auto-detected types
        je.set_detected_types({
            "set": set,
            "frozenset": frozenset,
        })
        out = json.loads(json.dumps(set_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(set_, out)
        out = json.loads(json.dumps(frozenset_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(frozenset_, out)
        out = json.loads(json.dumps(np_int64, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(np_int64, out)
        out = json.loads(json.dumps(np_float64, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(np_float64, out)
        out = json.loads(json.dumps(np_bool, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(np_bool, out)
        out = json.loads(json.dumps(np_datetime64, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(np_datetime64, out)
        out = json.loads(json.dumps(np_timedelta64, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(np_timedelta64, out)
        out = json.loads(json.dumps(pd_series, cls=JSONEncoder), cls=JSONDecoder)
        self.assertTrue(pd_series.equals(out))
        out = json.loads(json.dumps(pd_dataframe, cls=JSONEncoder), cls=JSONDecoder)
        self.assertTrue(pd_dataframe.equals(out))
        out = json.loads(json.dumps(dt, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(dt, out)
        out = json.loads(json.dumps(td, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(td, out)
        out = json.loads(json.dumps(date_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(date_, out)
        out = json.loads(json.dumps(time_, cls=JSONEncoder), cls=JSONDecoder)
        self.assertEqual(time_, out)


    def test_complex_types(self):
        je.DETECTED_TYPES = {}
        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3
                self.serie = pd.Series([1, 2, 3])

            def __eq__(self, other):
                return self.a == other.a and self.b == other.b and self.c == other.c and self.serie.equals(other.serie)

        my_class = MyClass()
        json_data = json.dumps(my_class, cls=JSONEncoder)
        out = json.loads(json_data, cls=JSONDecoder)
        self.assertEqual(my_class, out)

    def test_complex_call__json__(self):
        class MyClassJson:
            def __init__(self):
                self.a = 1
                self.b = 2
                self.c = 3
                self.serie = pd.Series([1, 2, 3])

            def __tojson__(self):
                return {"Hello": "World", "a": self.a, "b": self.b, "c": self.c, "serie": self.serie.to_dict()}

            @classmethod
            def __fromjson__(cls, data):
                my_class = cls.__new__(cls)
                my_class.a = data["a"]
                my_class.b = data["b"]
                my_class.c = data["c"]
                my_class.serie = pd.Series(data["serie"].values(), index=range(len(data["serie"])))
                return my_class


            def __eq__(self, other):
                a_eq = self.a == other.a
                b_eq = self.b == other.b
                c_eq = self.c == other.c
                s_eq = self.serie.equals(other.serie)
                return a_eq and b_eq and c_eq and s_eq

        my_class = MyClassJson()
        je.add_types(MyClassJson=MyClassJson)
        json_data = json.dumps(my_class, cls=JSONEncoder)
        out = json.loads(json_data, cls=JSONDecoder)
        self.assertEqual(my_class, out)
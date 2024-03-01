from unittest import TestCase
from src.backtest.engine import Metadata
from datetime import timedelta
from copy import deepcopy
from src.backtest.engine import Strategy
import subprocess

try:
    GIT_AUTHOR = subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").strip()
except:
    GIT_AUTHOR = "Unknown"
try:
    GIT_COMMIT = subprocess.check_output(["git", "log", "-n", "1"]).decode("utf-8").rstrip()
except:
    GIT_COMMIT = "Unknown"

class MyStrategy(Strategy):
    """
    This is a test strategy
    """
    def run(self, data, timestep):
        pass

class TestMetadata(TestCase):
    """
    Since the metadata class takes a lot of information from the environement, to ensure reproducable results of the
    test, we will only test if method works and do not throw.
    """

    def test___init__(self):
        """
        Test the __init__ method
        """
        metadata = Metadata(strategy_name="test", description="test", author="test", version="test", time_res=timedelta(seconds=1),
                            save_code=True, hash_only=True, file_blacklist=[])
        self.assertEqual(metadata.strategy_name, "test")
        self.assertEqual(metadata.description, "test")
        self.assertEqual(metadata.author, "test")
        self.assertEqual(metadata.version, "test")
        self.assertEqual(metadata.time_res, timedelta(seconds=1).total_seconds())
        self.assertTrue(metadata.save_code)
        self.assertTrue(metadata.hash_only)
        self.assertEqual(metadata.file_blacklist, [])
        metadata = Metadata()
        self.assertIsNone(metadata.strategy_name)
        self.assertIsNone(metadata.description)

        # ----------------------------------------------
        # Assuming you have a git repository
        self.assertNotEqual(metadata.author, "Unknown")
        self.assertNotEqual(metadata.version, "Unknown")
        # ----------------------------------------------
        self.assertIsNone(metadata.time_res)
        self.assertTrue(metadata.save_code)
        self.assertTrue(metadata.hash_only)
        self.assertEqual(metadata.file_blacklist, tuple())
        self.assertEqual(metadata.code_path, './**/*.py')

        metadata = Metadata(code_path="engine_tests/test_code/")
        self.assertEqual(metadata.code_path, 'engine_tests/test_code/*.py')
        metadata = Metadata(code_path="engine_tests/test_code")
        self.assertEqual(metadata.code_path, 'engine_tests/test_code/*.py')
        metadata = Metadata(code_path="engine_tests/test_code/**/*.txt")
        self.assertEqual(metadata.code_path, 'engine_tests/test_code/**/*.txt')


    def test_load_code(self):
        metadata = Metadata()
        expected_hash = {'test_code/file2.py': {'checksum': 'ef92f38fe19bfee5c26382059ed32bf4', 'code': None},
                         'test_code/file1.py': {'checksum': '57847040a18858dc222ba34f34ecd7c7', 'code': None}}

        files_hash = metadata.load_code(checksum=True, path="test_code/*.py")
        file1 = '"""\nThis is a file deisgned to test code extarction.  It is not meant to be run.\n"""\n\nimport unittest\n\nprint("Running file1.py")'
        file2 = ('"""\nThis is a file deisgned to test code extarction.  It is not meant to be run.\n"""\nfrom enum import '
                 'Enum\n\nclass TestObj(Enum):\n    A = \'A\'\n    B = \'B\'\n    C = \'C\'\n\n    def __str__(self):\n'
                 '        return self.value\n\n    @classmethod\n    def from_str(cls, string: str):\n        if string '
                 '== \'A\':\n            return cls.A\n        elif string == \'B\':\n            return cls.B\n        '
                 'elif string == \'C\':\n            return cls.C\n        else:\n            raise ValueError'
                 '(f"Invalid string: {string}")\n\n\nprint("TestObj.A: ", TestObj.A)\nprint("TestObj.A.value: ", '
                 'TestObj.A.value)\nprint("TestObj.A.name: ", TestObj.A.name)\nprint("TestObj.A.__str__(): ", '
                 'TestObj.A.__str__())\nprint("TestObj.from_str(\'A\'): ", TestObj.from_str(\'A\'))\n\n'
                 '# ----------------------------------------------\n\nprint("TestObj.B: ", TestObj.B)\n'
                 'print("TestObj.B.value: ", TestObj.B.value)\nprint("TestObj.B.name: ", TestObj.B.name)\n'
                 'print("TestObj.B.__str__(): ", TestObj.B.__str__())\nprint("TestObj.from_str(\'B\'): ", '
                 'TestObj.from_str(\'B\'))\n\n# ----------------------------------------------\n\n'
                 'print("TestObj.C: ", TestObj.C)\nprint("TestObj.C.value: ", TestObj.C.value)\n'
                 'print("TestObj.C.name: ", TestObj.C.name)\nprint("TestObj.C.__str__(): ", TestObj.C.__str__())\n'
                 'print("TestObj.from_str(\'C\'): ", TestObj.from_str(\'C\'))\n\n'
                 '# ----------------------------------------------\n\ntry:\n    '
                 'TestObj.from_str(\'D\')\nexcept Exception as e:\n    '
                 'print(f"Unable to load type from \'D\' (Which is normal)")\n    print(e)')

        expected_code = deepcopy(expected_hash)
        expected_code['test_code/file1.py']['code'] = file1
        expected_code['test_code/file2.py']['code'] = file2
        self.assertEqual(files_hash, expected_hash)
        files = metadata.load_code(checksum=False, path="test_code/*.py")
        self.assertEqual(files, expected_code)



    def test_init(self):
        """
        Test the init method
        """
        metadata = Metadata()
        self.assertRaises(ValueError, metadata.init, None, None, None, None, None)
        metadata.init(strategy=MyStrategy())
        self.assertEqual(metadata.strategy_name, "MyStrategy")
        self.assertEqual(metadata.description, "\n    This is a test strategy\n    ")

    def test_export(self):
        """
        Test the export method
        """
        metadata = Metadata(code_path="./test_code/", file_blacklist=["./test_code/file2.py"])
        metadata.init(strategy=MyStrategy(), run_duration=10.)

        self.assertEqual(metadata.export(), {'strategy_name': 'MyStrategy',
                                             'description': '\n    This is a test strategy\n    ',
                                             'author': GIT_AUTHOR,
                                             'version': GIT_COMMIT,
                                             'time_res': None,
                                             'save_code': True,
                                             'hash_only': True,
                                             'code': {'./test_code/file1.py': {'checksum': '57847040a18858dc222ba34f34ecd7c7', 'code': None}},
                                             'backtest_parameters': None,
                                             'tickers': None,
                                             'features': None,
                                             'run_duration': 10.})

    def test_load(self):
        state = {'strategy_name': 'MyStrategy',
                                             'description': '\n    This is a test strategy\n    ',
                                             'author': GIT_AUTHOR,
                                             'version': GIT_COMMIT,
                                             'time_res': None,
                                             'save_code': True,
                                             'hash_only': True,
                                             'code': {'./test_code/file1.py': {'checksum': '57847040a18858dc222ba34f34ecd7c7', 'code': None}},
                                             'backtest_parameters': None,
                                             'tickers': None,
                                             'features': None,
                                             'run_duration': 10.}

        self.assertEqual(Metadata.load(state).export(), state)




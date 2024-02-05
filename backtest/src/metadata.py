from typing import List, Optional, Iterable, Dict
from datetime import timedelta
from .strategy import Strategy
import subprocess
import os
import glob
import hashlib

class Metadata:
    """
    This class is meant to contain all the information needed to reproduce the results of the backtest of a startegy
    Usually, the dataset must be big, so it is recomended to write in the description how to acquire the dataset,or
    make a script that download it and save the script.
    """
    def __init__(self, strategy_name: Optional[str] = None, description: Optional[str] = None,
                 author: Optional[str] = None, version: Optional[str] = None, time_res: Optional[timedelta] = None,
                 save_code: bool = True, hash_only: bool = True, file_blacklist: Optional[Iterable[str]] = tuple(),
                 code_path: Optional[str] = None):
        """
        This class contains the metadata of the strategy.
        :param strategy_name: The name of the strategy.  If None, the name of the class will be used.
        :param description: The description of the strategy.  It is strongly recommended to write steps on how to
                            reproduce the results here.  If description is None, the docstring of the strategy will be
                            used.
        :param author: The author of the strategy.  If None, the git author will be used.
        :param version: The version of the strategy.  If None, the git commit hash will be used.  If no git
                        repository, it will be "Unknown"
        :param time_res: The time resolution.  If None, the automatically found time resolution will be saved. It is
                         the task of the broker to provide the time resolution.  (Not handled in this class)
        :param save_code: Whether to save the code of the strategy.  If True, the code will be saved.
        :param hash_only: Only used if save_code is True.  If True, only the hash of the code will be saved.
                            (It can spare storage space)
        :param file_blacklist: Files to not include in the code saved.  It is a list of strings with the file paths
                                relative to the current working directory.
        :param code_path: The path to the root folder where to search for the code.  Default is the current working
                            directory (If None).  If no '*' is present in the path, it will be added at the end to
                            search for py files.
        """
        self.strategy_name = strategy_name
        self.description = description
        self.author = author if author is not None else self.get_git_author()
        self.version = version if version is not None else self.get_git_commit()
        # The time resolution is saved in seconds
        self.time_res = time_res.total_seconds() if time_res is not None else None
        self.save_code = save_code
        self.hash_only = hash_only
        self.file_blacklist = file_blacklist
        self.code = None
        self.code_path = code_path if code_path is not None else './**/*.py'
        if '*' not in self.code_path:
            self.code_path = self.code_path.rstrip("/") + "/*.py"

    def init(self, strategy: Strategy, backtest_parameters: Optional[dict] = None, tickers: Optional[List[str]] = None,
             features: Optional[List[str]] = None, run_duration: Optional[float] = None):
        """
        This method is used to initialize the metadata object from the Backtest object (When the simulation has started)
        :param strategy: The strategy object.  (Used to get its name
        :param backtest_parameters: The parameters of the backtest object (Init paraeters)
        :param tickers: The tickers used in the backtest
        :param features: The features used in the backtest (Columns)
        :param run_duration: The duration of the backtest (in seconds)
        :return: None
        """
        if strategy is None:
            raise ValueError("strategy cannot be None")
        if self.description is None:
            self.description = strategy.__doc__
        if self.strategy_name is None:
            self.strategy_name = strategy.__class__.__name__
        self.run_duration = run_duration
        self.backtest_parameters = backtest_parameters
        self.tickers = tickers
        self.features = features
        if self.save_code:
            self.code = self.load_code(checksum=self.hash_only, path=self.code_path, ignore=self.file_blacklist)
        else:
            self.code = None

    def __str__(self):
        return f"Strategy Name: {self.name}\n\tDescription: {self.description}\n\tAuthor: {self.author}\n\tVersion: {self.version}"

    @staticmethod
    def get_git_author() -> str:
        """
        This method return the git author name.
        :return: git author name
        """
        try:
            return subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").strip()
        except:
            return "Unknown"

    def load_code(self, checksum: bool = True, ignore: Iterable[str] = tuple(), path: str = f"./**/*.py") \
            -> Dict[str, Dict[str, Optional[str]]]:
        """
        This method load the code of the strategy.
        :param path: The path to search for the code.  Default is the current working directory. (And ignore venv)
        :param checksum: Whether to return only the checksum of each file.
        :param ignore: Files to ignore when loading the code.
        :return: The code of the strategy
        """
        paths = glob.glob(path, recursive=True)
        paths_filtered = [path for path in paths if "/venv/" not in path and path not in ignore]
        files = {}
        for path in paths_filtered:
            hash_md5 = hashlib.md5()
            with open(path, "r") as file:
                content = file.read()
                hash_md5.update(content.encode("utf-8"))
            if checksum:
                files[path] = {
                    "checksum": hash_md5.hexdigest(),
                    "code": None
                }
            else:
                files[path] = {
                    "checksum": hash_md5.hexdigest(),
                    "code": content
                }

        return files

    def get_git_commit(self) -> str:
        """
        This method return the git commit hash.
        :return: git commit hash
        """
        try:
            return subprocess.check_output(["git", "log", "-n", "1"]).decode("utf-8").rstrip()
        except:
            return "Unknown"

    def export(self) -> dict:
        """
        This method export the metadata to a JSONable dictionary.
        Note:
            The file_blacklist parameter is not saved (and not loaded) for privacy purposes.
        :return: The state of the object as a dictionary
        """
        return {
            "strategy_name": self.strategy_name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "time_res": self.time_res,
            "save_code": self.save_code,
            "hash_only": self.hash_only,
            "code": self.code,
            "backtest_parameters": self.backtest_parameters,
            "tickers": self.tickers,
            "features": self.features,
            "run_duration": self.run_duration
        }

    @classmethod
    def load(cls, data: dict):
        """
        This method load the metadata from a dictionary.
        NOTE:
            The file_blacklist parameter is not loaded (and not saved) for privacy purposes.
        :param data: The dictionary to load the metadata from.
        :return: The metadata object
        """
        metadata = cls()
        metadata.strategy_name = data["strategy_name"]
        metadata.description = data["description"]
        metadata.author = data["author"]
        metadata.version = data["version"]
        metadata.time_res = data["time_res"]
        metadata.save_code = data["save_code"]
        metadata.hash_only = data["hash_only"]
        metadata.code = data["code"]
        metadata.backtest_parameters = data["backtest_parameters"]
        metadata.tickers = data["tickers"]
        metadata.features = data["features"]
        metadata.run_duration = data["run_duration"]
        return metadata
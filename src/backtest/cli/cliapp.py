"""
Copyright (C) 2024 Anthony Lavertu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Implement the base class for a CliApp.
"""
from abc import ABC, abstractmethod
from typing import List, Type, Set, Iterable
import sys
from enum import Enum

COLOR = "\033[38;5;203m"
RESET = "\033[0m"

class ErrorType(Enum):
    InvalidArgumentError = "InvalidArgumentError"
    InvalidArgumentTypeError = "InvalidArgumentTypeError"
    UnexpectedError = "UnexpectedError"

    def __str__(self):
        return self.value


def Error(message: str, T: ErrorType):
    print(f"{COLOR}{T}: {message}{RESET}", file=sys.stderr)
    exit(-1)

def indent_lines(text: str, indent: int) -> str:
    return "\n".join([" " * indent + line for line in text.split("\n")])

class CliApp(ABC):
    def __init__(self, args: List[Type], error_msg: str = "", available_flags: Iterable[str] = tuple(), **kwargs: Type):
        self.kwargs = kwargs
        self.args = args
        self.error_msg = error_msg
        self.available_flags = set(available_flags)

    @abstractmethod
    def run(self, flags: List[str], *args, **kwargs) -> None:
        pass


    def exec(self, flags: List[str], *args, **kwargs) -> None:
        # Early exit if the user only wanted help
        if len(args) > 0 and (args[0].strip("-") == "help" or args[0].strip("-") == "h"):
            print(self.help())
            return
        self.check(flags, args, kwargs)
        self.run(flags, *args, **kwargs)


    @abstractmethod
    def help(self) -> str:
        pass

    def check(self, flags: List[str], args, kwargs):
        # Verify args
        if len(args) < len(self.args):
            Error(f"{self.error_msg}\n{indent_lines(f'There are not enough arguments passed. '
                                                    f'Expected: {len(self.args)}', 4)}, but got {len(args)}.",
                  ErrorType.InvalidArgumentError)
        elif len(args) > len(self.args):
            Error(f"{self.error_msg}\n{indent_lines('There are too many arguments passed. '
                                                    f'Expected: {len(self.args)}', 4)}",
                  ErrorType.InvalidArgumentError)

        for i, arg in enumerate(args):
            if not isinstance(arg, self.args[i]):
                Error(f"{self.error_msg}\n{indent_lines(f'Argument at index {i} is of the wrong type.  '
                                                        f'Got {arg} of type {arg.__class__.__name__}, but expected '
                                                        f'type is {self.args[i].__class__.__name__}', 4)}",
                      ErrorType.InvalidArgumentTypeError)

        # Verify kwargs
        for key, value in kwargs.items():
            if key not in self.kwargs:
                Error(f"{self.error_msg}\n{indent_lines(f'Unexpected keyword argument {key}.', 4)}",
                      ErrorType.InvalidArgumentError)
            if not isinstance(value, self.kwargs[key]):
                Error(f"{self.error_msg}\n{indent_lines(f'Keyword argument {key} is of the wrong type.  '
                                                        f'Got {value} of type {value.__class__.__name__}, but expected '
                                                        f'type is {self.kwargs[key].__class__.__name__}', 4)}",
                      ErrorType.InvalidArgumentTypeError)

        # Verify flags
        for flag in flags:
            if flag not in self.available_flags:
                Error(f"{self.error_msg}\n{indent_lines(f'Unexpected flag {flag}.', 4)}",
                      ErrorType.InvalidArgumentError)


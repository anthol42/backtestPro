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

This file is the entry point to the cli interface.  It will route to the appropriate subcommand. (CliApp)
"""
from cli import CliApp, Error, ErrorType, Show, Template
from typing import List, Dict, Tuple, Union
import sys
from pathlib import PurePath
from __version__ import __version__
class App:
    def __init__(self, apps: Dict[str, CliApp]):
        self.apps = apps
        self.integrated = {
            "version": self.version,
            "help": self.help,
            "v": self.version,
            "h": self.help
        }

    def version(self):
        with open(PurePath(__file__).parent / "version.txt", "r") as f:
            version_txt = f.read()
        version_txt = version_txt.format(__version__)
        print(version_txt + "\n")

    def help(self):
        with open(PurePath(__file__).parent / "help.txt", "r") as f:
            help_txt = f.read()
        print(help_txt + "\n")

    def run(self):
        if len(sys.argv) < 2:
            Error("No subcommand provided.", ErrorType.InvalidArgumentError)

        subcommand = sys.argv[1]
        args, flags, kwargs = self.parse_command(sys.argv[2:])
        if subcommand in self.apps:
            app = self.apps[subcommand]
            app.exec(flags, *args, **kwargs)
        elif subcommand.strip("-") in self.integrated:
            self.integrated[subcommand.strip("-")]()
        else:
            Error(f"Subcommand '{subcommand}' not found.", ErrorType.InvalidArgumentError)


    def build_str(self, args: List[str]):
        s = ""
        for arg in args:
            if " " in arg:
                s += f'"{arg}" '
            else:
                s += f"{arg} "
        return s

    @staticmethod
    def cast(value: str) -> Union[str, int, float, bool, None]:
        """
        Convert to the appropriate basic python type
        :param value: The value to cast
        :return: The casted value
        """
        if value.lower() == "none" or value.lower() == "null":
            return None
        elif value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def parse_command(self, command: List[str]) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Convert a command string into args and kwargs and flags.
        Syntax: subcommand arg1 arg2 -flag1 -flag2 --kwarg1=kwarg1_value --kwarg2 kwarg2_value
        Every parameter that doesn't start with - is considered an arg.
        Every parameter that starts with - is considered a flag and cannot have a value associated with it.
        Every parameter that starts with -- is considered a kwarg and must have a value associated with it.
        Its value can be separated by = or by a space.
        :param command: The command string to parse.
        :return: Arguments, flags, and keyword arguments.
        """
        kwargs = {}
        flags = []
        args = []

        skip = False
        for i, arg in enumerate(command):
            if skip:
                skip = False
                continue
            if arg.startswith("--"):
                if "=" in arg:
                    s = arg.split("=")
                    key = s[0][2:]
                    value = "=".join(s[1:])
                    kwargs[key] = self.cast(value)
                else:
                    if i + 1 >= len(command):
                        Error(f"Expected a value for keyword argument {arg}.", ErrorType.InvalidArgumentError)
                    kwargs[arg[2:]] = self.cast(command[i + 1])
                    skip = True
            elif arg.startswith("-"):
                flags.append(arg[1:])
            else:
                args.append(self.cast(arg))

        return args, flags, kwargs




if __name__ == "__main__":
    apps = {
        "show": Show(),
        "init": Template(),
    }
    App(apps).run()



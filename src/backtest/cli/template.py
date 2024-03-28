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

This file implements the utility functions to built the template for a backtest project.
"""
from .cliapp import CliApp, Error, ErrorType
from pathlib import PurePath
import os
import shutil
import subprocess

class Template(CliApp):
    def __init__(self):
        super().__init__([str], available_flags=["g"])

    def run(self, flags, *args, **kwargs):
        path = args[0]
        absolute_path = PurePath(os.getcwd()) / path
        if not os.path.exists(absolute_path.parent):
            Error(f"Path '{absolute_path.parent}' does not exist.", ErrorType.InvalidArgumentError)

        shutil.copytree(PurePath(__file__).parent / "template", absolute_path)
        if "g" in flags:
            self.init_git(absolute_path)
        print(f"Project created at {absolute_path}")

    def help(self):
        return """
    Create a new backtest project at the specified path.
    
    Usage: backtest template [[flags]] [path]
    
    Where:
        [path] - The path to create the project at.
        [flags] - Optional flags to modify the behavior of the command.
        
        Flags:
            -g - Initialize a git repository in the project.
        """


    def init_git(self, path: str):
        subprocess.run(["git", "init"], cwd=path)

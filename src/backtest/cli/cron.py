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
"""This file implements the utility functions to schedule a cron job from the cli."""
from .cliapp import CliApp, Error, ErrorType
from typing import List, Dict, Tuple, Union, Optional
import os
from crontab import CronTab
from pathlib import PurePath
import sys

BACKTEST_TAG = "bcktst-pro: "


class Cron(CliApp):
    def __init__(self):
        super().__init__([str], path=str, name=str, m=(str, int), h=(str, int), dom=(str, int), mon=(str, int),
                         dow=(str, int), logfile=str)


    def verify_cron(self, value: str, idx: int, r: Tuple[int, int]) -> bool:
        """
        Verify that a cron expression is valid.
        :param value: The cron expression to verify
        :param idx: The index of the cron expression
        :param r: The range of the cron expression
        :return: None
        """
        if isinstance(value, int):
            value = str(value)

        # Check if only wildcard
        if value == "*":
            return True

        # Check if the value is a number
        if value.isdigit() and r[0] <= int(value) <= r[1]:
            return True

        # Check if the value is a range
        if "-" in value:
            s = value.split("-")
            if len(s) != 2:
                Error(f"Invalid range expression at index {idx}", ErrorType.InvalidArgumentError)
            start, end = s
            if start.isdigit() and end.isdigit():
                if r[0] <= int(start) <= int(end) <= r[1]:
                    return True

        # Check if the value is a step
        if "/" in value:
            s = value.split("/")
            if len(s) != 2:
                Error(f"Invalid step expression at index {idx}", ErrorType.InvalidArgumentError)
            value, step = s
            if (value.isdigit() or value == "*") and step.isdigit():
                if r[0] <= int(step) <= r[1]:
                    if value.isdigit():
                        if r[0] <= int(value) <= r[1]:
                            return True
                    else:
                        return True
                return False

        # Check if the value is a list
        if "," in value:
            s = value.split(",")
            for v in s:
                if not self.verify_cron(v, idx, r):
                    return False
            return True
        return False

    def add(self, path: str, name: str, m: str = "*", h: str = "*", dom: str = "*", mon: str = "*", dow: str = "*",
            logfile: Optional[str] = None):
        """
        Add a cron job to the system.
        Args:
            path: The path to the file to run (The job)
            name: The name of the job (Must be unique)
            m: The minute expression
            h: The Hour expression
            dom: The Day of Month expression
            mon: The month expression
            dow: The Day of Week expression
            logfile: The path to the log file

        Returns:

        """

        if not os.path.isabs(path):
            path = PurePath(os.getcwd()) / path
        if not os.path.exists(path):
            Error(f"Path '{path}' does not exist.", ErrorType.InvalidArgumentError)

        # Verify minute
        if not self.verify_cron(m, 0, (0, 59)):
            Error(f"Invalid minute expression '{m}'", ErrorType.InvalidArgumentError)

        # Verify hour
        if not self.verify_cron(h, 1, (0, 23)):
            Error(f"Invalid hour expression '{h}'", ErrorType.InvalidArgumentError)

        # Verify day of month
        if not self.verify_cron(dom, 2, (1, 31)):
            Error(f"Invalid day of month expression '{dom}'", ErrorType.InvalidArgumentError)

        # Verify month
        if not self.verify_cron(mon, 3, (1, 12)):
            Error(f"Invalid month expression '{mon}'", ErrorType.InvalidArgumentError)

        # Verify day of week
        if not self.verify_cron(dow, 4, (0, 6)):
            Error(f"Invalid day of week expression '{dow}'", ErrorType.InvalidArgumentError)

        cron = CronTab(user=True)
        ids = {job.comment for job in cron}
        if BACKTEST_TAG + name in ids:
            Error(f"Job with name '{name}' already exists.", ErrorType.InvalidArgumentError)

        python_path = sys.executable
        if not os.path.isabs(path):
            path = PurePath(os.getcwd()) / path
        if logfile:
            if not os.path.isabs(logfile):
                logfile = PurePath(os.getcwd()) / logfile
            job = cron.new(command=f'{python_path} {path} > {logfile} 2>&1',
                           comment=BACKTEST_TAG + name)
        else:
            job = cron.new(command=f'{python_path} {path}',
                           comment=BACKTEST_TAG + name)

        job.setall(f"{m} {h} {dom} {mon} {dow}")

        cron.write()

        print("Job added successfully.")

    def ls(self):
        """
        List all the cron jobs created by this program.
        :return: None
        """
        cron = CronTab(user=True)
        table: List[Tuple[str, str]] = []    # List of tuples (slices, comment)
        for job in cron:
            if job.comment.startswith(BACKTEST_TAG):
                table.append((str(job.slices), job.comment[len(BACKTEST_TAG):]))

        if len(table) == 0:
            print("No jobs found.")
            return
        # Get max slice length
        max_slice = max(len(s) for s, _ in table)
        max_name = max(len(n) for _, n in table)
        padding = 2

        # Header
        print("Schedule".ljust(max_slice + padding - 1), "│", "Name".ljust(max_name + padding))
        print("-" * (max_slice + max_name + padding + padding + 1))
        for slices, name in table:
            print(f"{slices.ljust(max_slice + padding)}│{' ' * padding}{name.ljust(max_name + padding)}")


    def delete(self, name: str):
        """
        Delete a cron job by name.
        :param name: The name of the job to delete
        :return: None
        """
        cron = CronTab(user=True)
        for job in cron:
            if job.comment == BACKTEST_TAG + name:
                cron.remove(job)
                cron.write()
                print(f"Job '{name}' deleted.")
                return
        Error(f"Job '{name}' not found.", ErrorType.InvalidArgumentError)

    def run(self, flags: List[str], *args, **kwargs) -> None:
        command = args[0]
        if command == "add":
            if "path" not in kwargs:
                Error("Missing required argument 'path'.", ErrorType.InvalidArgumentError)
            if "name" not in kwargs:
                Error("Missing required argument 'name'.", ErrorType.InvalidArgumentError)
            self.add(**kwargs)
        elif command == "ls":
            self.ls()
        elif command == "delete":
            if "name" not in kwargs:
                Error("Missing required argument 'name'.", ErrorType.InvalidArgumentError)
            self.delete(**kwargs)
        else:
            Error(f"Command '{command}' not found.", ErrorType.InvalidArgumentError)

    def help(self) -> str:
        return """
    Manage cron jobs on systems where it is available.
    
    Syntax: backtest cron [command] [args]
    
    Where:
        [command] - The command to run.
        [args] - The arguments to the command.
    
    Commands:
        
    - add - Add a new cron job.
        Parameters: 
            path [REQUIRED]: The path to the file to run.
            name [REQUIRED]: The name of the job.
            m [OPTIONAL]: The minute expression.
            h [OPTIONAL]: The hour expression.
            dom [OPTIONAL]: The day of month expression.
            mon [OPTIONAL]: The month expression.
            dow [OPTIONAL]: The day of week expression.
            logfile [OPTIONAL]: The path to the log file.
    
    - delete - Delete a cron job.
        Parameters:
            name [REQUIRED]: The name of the job to delete.
    
    - ls - List all the cron jobs created by this program.
    
    Examples:
        backtest cron add --path=./job.py name=test --logfile=log.txt --m=*/2
        backtest cron ls
        backtest cron delete --name=test
        """


if __name__ == "__main__":
    c = Cron()
    # c.add("../../../tmp2.py", "test2.0", logfile="log.txt", m="*/2")
    c.ls()
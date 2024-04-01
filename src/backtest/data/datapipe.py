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
from abc import ABC
from typing import List, Union, Optional, Any, Iterator, TypeVar, Generic
from enum import Enum
from datetime import datetime
import os
import pickle
import hashlib
import types
import inspect
import shutil


def clear_cache():
    if os.path.exists(".cache"):
        shutil.rmtree(".cache")

    DataPipe.LAST_ID = 0

def toHash(obj) -> str:
    """
    Hash an object.  The hash is consistent across runs and is deterministic given the object.
    Note:
        This function doesn't support all types of objects.
        This function is recursive and will hash the object recursively.
    :param obj: The object to has
    :return: The object hash
    """
    if isinstance(obj, dict):
        return toHash(str({k: toHash(v) for k, v in obj.items()}))
    elif isinstance(obj, str):
        hash_object = hashlib.sha256()
        hash_object.update(obj.encode())
        return hash_object.hexdigest()
    elif isinstance(obj, types.FunctionType):
        return toHash(str({"name": obj.__name__, "type": "Function",  "code": inspect.getsource(obj)}))
    elif hasattr(obj, "__dict__"):
        return toHash(obj.__dict__)
    elif hasattr(obj, "__iter__"):
        return toHash(str([toHash(o) for o in obj]))
    else:
        return toHash(str(obj))

class CacheObject:
    """
    This class holds a the output of a pipe and some metadata to be able to revalidate the cache.
    """
    def __init__(self, value: Any, pipe_id: int, pipe_hash: str, next_revalidate: Optional[datetime] = None,
                 max_request: Optional[int] = None, current_n_requests: int = 0):
        """
        :param value: The output of the pipe
        :param pipe_id: The pipe_id of the pipe that generated the cache
        :param pipe_hash: Used to check if the pipe structure has changed
        :param next_revalidate: The next time the cache should be revalidated
        :param max_request: The maximum number of requests that can be made before revalidating the cache
        :param current_n_requests: The current number of requests made to the cache
        """
        self.value = value
        self.pipe_id = pipe_id
        self.pipe_hash = pipe_hash
        self.write_time = datetime.now()
        self.next_revalidate = next_revalidate
        self.max_request = max_request
        self.current_n_requests = current_n_requests

    def store(self):
        """
        Call this method to store the cache to the .cache folder.  The cache file has the pipe_id as its name.  This
        is why it is strongly recommended to delete the .cache folder when changing the pipeline structure, to avoid
        unexpected behavior.
        """
        if not os.path.exists(".cache"):
            os.makedirs(".cache")
        with open(f".cache/{self.pipe_id}.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, pipe_id: int) -> 'CacheObject':
        """
        Call this method to load a cache from the .cache folder.
        :param pipe_id: The pipe_id that stored the cache
        :return: The cache object if it exists else None.  (This function doesn't check if the cache valid)
        """
        with open(f".cache/{pipe_id}.pkl", "rb") as file:
            return pickle.load(file)


class DataPipeType(Enum):
    """
    [Enum] - An enumeration representing the four types of pipes that can be used in the pipeline.

    - FETCH: Fetch data from an external source (Web, DB, file, API, etc.).
    - PROCESS: Process and transform the data (Filter, transform, etc.).
    - CACHE: Cache the output of the pipeline to avoid repeating previous steps on each run.
    - COLLATE: Collate the output of two pipes (This is a special case, used to merge two pipelines).
    """
    # Fetch data from an external source (Web, DB, file, API, etc.).
    FETCH = "FETCH"

    # Process and transform the data (Filter, transform, etc.).
    PROCESS = "PROCESS"

    # Cache the output of the pipeline to avoid repeating previous steps on each run.
    CACHE = "CACHE"

    # Collate the output of two pipes (This is a special case, used to merge two pipelines).
    COLLATE = "COLLATE"


T_ = TypeVar("T_")


class PipeOutput(Generic[T_]):
    """
    This class holds the output of a pipe and some metadata that are passed through out the pipeline.
    This metadata is used for revalidation purpose.
    """
    def __init__(self, value: T_, output_from: 'DataPipe', revalidate_action: Optional['RevalidateAction'] = None,
                 **kwargs):
        """
        Args:
            value: The value
            output_from: The output from the pipe
            revalidate_action: The revalidate action
            **kwargs:
        """
        if "value" in kwargs:
            raise ValueError("value is a reserved keyword, you cannot use it as a keyword argument in PipeOutput.__init__")
        if "output_from" in kwargs:
            raise ValueError("output_from is a reserved keyword, you cannot use it as a keyword argument in PipeOutput.__init__")
        if "revalidate_action" in kwargs:
            raise ValueError("revalidate_action is a reserved keyword, you cannot use it as a keyword argument in PipeOutput.__init__")
        self._value = value
        self._output_from = output_from
        self._revalidate_action = revalidate_action
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_revalidate(self, action: 'RevalidateAction'):
        """
        Set the revalidate action of the pipe.
        :param action: The action to set
        :return: None
        """
        self._revalidate_action = action

    def set_output_from(self, output_from: 'DataPipe'):
        """
        Set the output_from of the pipe.  Can be useful when a pip makes a copy of the output of another pipe.
        :param output_from: The pipe that generated the output
        :return: None
        """
        self._output_from = output_from

    @property
    def value(self):
        """
        The output of the pipe
        :return: The output value of the pipe
        """
        return self._value

    @property
    def revalidate(self):
        """
        Get the revalidate action of the pipe
        :return: The revalidate action of the pipe
        """
        return self._revalidate_action

    def __str__(self):
        return f"PipeOutput({repr(self.value)}, from={repr(self._output_from)})"


class RevalidateAction(Enum):
    """
    The three possible revalidation action that a pipe can return.

    - REVALIDATE: Revalidate the cache from the current position to the end of the pipeline
    - FULL_REVALIDATE: Revalidate the cache from the beginning to the end of the pipeline
    - NOOP: Do not revalidate the cache (The pipe will load from cache if it exists)
    """
    REVALIDATE = "REVALIDATE"
    FULL_REVALIDATE = "FULL_REVALIDATE"
    NOOP = "NOOP"

class DataPipe(ABC):
    """
    The base element of the data module api.  This object can represent the four types of pipes (Fetch, Process, Cache,
    and Collate).  It is a recursive object that can be used to build complex pipelines.  The pipeline is built using
    the pipe operator (|).  Once the pipeline is built, it is run by calling the get method.
    To make custom pipes, you need to inherit from this class and implement one of the three methods (fetch, process,
    collate). If you want to implement a cache pipe, you need to implement the load, cache and revalidate methods.
    Do not forget to call the super().__init__ method in the __init__ method of your custom pipe and pass the
    appropriate DataPipeType.  You can also pass a name to the pipe to make the pipeline more readable.
    """
    LAST_ID = 0
    def __init__(self, T: DataPipeType, name: Optional[str] = None):
        """
        :param T: The pipe type (Fetch, Process, Cache, Collate)
        :param name: The pipe name.  If None, the name of the class is used
        """
        self._pipes: Optional[Union[DataPipe, List[DataPipe]]] = None
        self.T = T
        self._cache: Optional[CacheObject] = None
        self.name = name if name is not None else self.__class__.__name__
        self._pipe_id: int = 0    # Ids are given at built time and are deterministic given the structure of the pipe.
        self._has_run = False


    def get(self, frm: datetime, to: datetime, *args, **kwargs) -> Any:
        """
        Run the pipeline from the beginning to the end and return the output of the pipeline.
        :param frm: From datetime, this is passed to all pipes and is used to fetch the data from this date to the 'to'
        date.
        :param to: The end datetime, this is passed to all pipes and is used to fetch the data from the 'frm' date to
        this date.
        :param args: any args that needs to be passed to the subsequent pipes
        :param kwargs: any keyword args that needs to be passed to the subsequent pipes
        :return: The output of the pipeline
        """
        # Step 0: Evaluate if we have the good id
        if not self._has_run:
            if DataPipe.LAST_ID != 0:
                self._increment_id(DataPipe.LAST_ID + 1)
            DataPipe.LAST_ID = self._pipe_id

        # Step1: Load cache from disk
        flatten: List[DataPipe] = []    # Will become an array of references to all the pipes in the pipeline (Might not be in order)
        self._flatten(flatten)
        for pipe in flatten:
            pipe._load_cache()
        # Step2: Run the pipeline
        out = self._run(frm, to, *args, po=None, **kwargs)
        if out is None:
            raise ValueError(f"Pipe {self.name} returned None")
        # We run again the pipeline without using cache data.
        if out.revalidate == RevalidateAction.FULL_REVALIDATE:
            out = self._run(frm, to, *args, po=None, force_reload=True, **kwargs)
        # Unwrap the output (We do not want to return a PipeOutput object)
        return out.value

    def _run(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Any]] = None, force_reload: bool = False,
             **kwargs) -> PipeOutput:
        """
        This method is called by the get method to run the pipeline.  It is also called by each pipe to run the next
        pipe.
        :param frm: The from datetime
        :param to: The to datetime
        :param args: Any args that needs to be passed to the subsequent pipes
        :param po: The previous output of the pipeline.  Can be None
        :param force_reload: Whether to force the revalidation of the cache.  If True, the cache will be revalidated
                            no matter what.
        :param kwargs: The keyword args that needs to be passed to the subsequent pipes
        :return: The output of the pipeline
        """
        self._has_run = True  # The pipe_id is now frozen.
        # Early exit if the pipe return a FULL_REVALIDATE action because we will need to revalidate everything
        if po is not None and po.revalidate == RevalidateAction.FULL_REVALIDATE:
            return po

        if self.T == DataPipeType.COLLATE:
            po1 = self._pipes[0]._run(frm, to, *args, po=po, force_reload=force_reload, **kwargs)
            po2 = self._pipes[1]._run(frm, to, *args, po=po, force_reload=force_reload, **kwargs)
            # Early exit if the pipe return a FULL_REVALIDATE action because we will need to revalidate everything
            if po1.revalidate == RevalidateAction.FULL_REVALIDATE or po2.revalidate == RevalidateAction.FULL_REVALIDATE:
                return PipeOutput(None, self, revalidate_action=RevalidateAction.FULL_REVALIDATE)
        elif self._pipes is not None and self.T != DataPipeType.CACHE:
            po = self._pipes._run(frm, to, *args, po=po, force_reload=force_reload, **kwargs)

        if self.T == DataPipeType.FETCH:
            rev = po.revalidate if po is not None else RevalidateAction.NOOP
            if rev != RevalidateAction.FULL_REVALIDATE:
                po = self.fetch(frm, to, *args, po=po, **kwargs)
                po.set_revalidate(rev)
        elif self.T == DataPipeType.PROCESS:
            rev = po.revalidate if po is not None else RevalidateAction.NOOP
            if rev != RevalidateAction.FULL_REVALIDATE:
                po = self.process(frm, to, *args, po=po, **kwargs)
                po.set_revalidate(rev)
        elif self.T == DataPipeType.COLLATE:
            po = self.collate(frm, to, *args, po1=po1, po2=po2, **kwargs)
            if po1.revalidate == RevalidateAction.FULL_REVALIDATE or po2.revalidate == RevalidateAction.FULL_REVALIDATE:
                po.set_revalidate(RevalidateAction.FULL_REVALIDATE)
        # Handle cache
        elif self.T == DataPipeType.CACHE:
            # If there is no cache or we force the revalidation (Likely triggered by a FULL_REVALIDATE action)
            if self._cache is None or force_reload:
                po = self._pipes._run(frm, to, *args, po=po, force_reload=force_reload, **kwargs)
                self.cache(frm, to, *args, po=po, **kwargs)
            # We verify if caching is up to date
            else:
                revalidate_action = self.revalidate(frm, to, *args,
                                             po=PipeOutput(po, output_from=self, cache=self._cache), **kwargs)
                # We make a full revalidation of the cache (Upstream revalidation)
                if revalidate_action == RevalidateAction.FULL_REVALIDATE:
                    po = PipeOutput(None, self, revalidate_action=RevalidateAction.FULL_REVALIDATE)
                # We revalidate the cache from our current position to the end of the pipeline.(Downstream revalidation)
                elif revalidate_action == RevalidateAction.REVALIDATE:
                    po = self._pipes._run(frm, to, *args, po=po, force_reload=force_reload, **kwargs)
                    po.set_revalidate(revalidate_action)
                    self.cache(frm, to, *args, po=po, **kwargs)
                # Cache is up to date, no need to run this section of the pipeline, so we return the cache.
                else:
                    po = PipeOutput(self._cache.value, self, revalidate_action=revalidate_action)
        else:
            raise NotImplementedError(f"DataPipeType {self.T} not implemented")
        return po

    def _build(self, other: 'DataPipe') -> 'DataPipe':
        """
        This method is used to build the pipeline from multiple pipes.  It is called when using the pipe operator (|).
        :param other: The other pipe (Right hand side of the pipe operator)
        :return: A new pipe that is the result of the concatenation of the two pipes.  (Multiple pipes makes a pipeline)
        """
        new = other
        new._pipes = self
        new._pipe_id = self._pipe_id + 1
        return new

    def __or__(self, other: 'DataPipe') -> 'DataPipe':
        """
        This is called the pipe operator in the case of this class.  It is used to build the pipeline from multiple
        pipes.
        :param other: The other pipe (Right hand side of the pipe operator)
        :return: A new pipe that is the result of the concatenation of the two pipes.  (Multiple pipes makes a pipeline)
        """
        return self._build(other)

    def hash(self) -> str:
        """
        Make a hash that is consistent across runs and is deterministic given the object.
        :return: The hash.
        """
        d = self.__dict__.copy()
        d.pop("_cache", None)    # Will change even if the structure didn't.
        d.pop("_pipes", None)    # Handled separately
        d.pop("_has_run", None)    # Will change even if the structure didn't.
        d.pop("_pipe_id", None)    # This is subject to change even though the structure didn't.
        if self._pipes is not None:
            if isinstance(self._pipes, list):
                pipe_hash = [p.hash() for pipeline in self._pipes for p in pipeline]
            else:
                pipe_hash = [p.hash() for p in self._pipes]
        else:
            pipe_hash = []
        hashs = str([toHash(o) for o in d.values()] + pipe_hash)
        return toHash(hashs)

    @classmethod
    def Collate(cls, pipe1: 'DataPipe', pipe2: 'DataPipe') -> 'DataPipe':
        """
        This method is used to build a collate pipe from two pipes.  It is used top merge two branches of a pipeline.
        :param pipe1: The first pipeline (or pipe if there is only one)
        :param pipe2:  The second pipeline (or pipe if there is only one)
        :return: A new pipe that is the result of the concatenation of the two pipes.  (Multiple pipes makes a pipeline)
        """
        new = cls(DataPipeType.COLLATE, name=f"Collate")
        new._pipes = [pipe1, pipe2]
        new._pipe_id = pipe1._pipe_id + pipe2._pipe_id + 2
        pipe2._increment_id(pipe1._pipe_id + 1)    # Increment the pipe_id of the second branch and all its children
        return new

    def set_id(self, pipe_id: int):
        """
        With this method, it is possible to manually set the pipe_id of the pipe.  It is recommended to call this method
        only on the top-level pipes, and before running the pipeline with get.  If the pipe is not the top-level pipe,
        its pipe_id may change, and you might run in unexpected behaviors.  By running this method only before running
        the get method, this ensures that the pipe will have the same pipe_id if the pipeline structure is the same.
        This is due to the fact that pipe_ids are dynamically assigned at build time.  However, when the pipe is run,
        the pipe becomes forged and no changes are allowed.  This ensures that the pipe ids stays the same.
        When this method is called, it acts as if the pipe was run, and the pipe_ids are frozen.  This means that the
        ids aren't verified.  Make sure that the pipe_id given are unique to avoid conflicts.
        Note: The passed pipe_id will be assigned to the lowes-level pipe.  The current pipe will have the highest
        pipe_id
        :param pipe_id: The pipe_id to assign to the pipe.
        :return: None
        """
        if self._has_run:
            raise RuntimeError("You cannot set the pipe_id of a pipe that has already been run.  Once a pipe is run, "
                               "its ids are frozen.")
        self._pipe_id = self._increment_id(pipe_id) - 1
        self._has_run = True

    def _increment_id(self, new_pipe_id: int) -> int:
        """
        This method is used to increment the pipe_id of the pipe and all its children.  It is used to avoid having two
        pipes with the same pipe_id in the pipeline.  It is called when building a collate pipe.
        :param new_pipe_id: The start id to increment from
        :return: the maximum pipe id used
        """
        flatten = []    # Will become an array of references to all the pipes in the pipeline (Might not be in order)
        self._flatten(flatten)
        for pipe in flatten:
            if not pipe._has_run:
                pipe._pipe_id = new_pipe_id
                new_pipe_id += 1
        return new_pipe_id

    def _flatten(self, flatten_pipe: List['DataPipe']):
        """
        Flatten the pipeline into a list of pipes.  This is used to access all the pipes in the pipeline.
        :param flatten_pipe: A list where each reference to a pipe will be appended
        :return: None
        """
        if self._pipes is not None:
            if isinstance(self._pipes, list):
                for pipe in self._pipes:
                    pipe._flatten(flatten_pipe)
            else:
                self._pipes._flatten(flatten_pipe)
        flatten_pipe.append(self)

    def _load_cache(self):
        """
        Load the cache from the disk.  This method is called by the get method to load the cache from the disk before
        running the pipeline.
        :return: None
        """
        if self.T == DataPipeType.CACHE:
            self._cache = self.load()

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput[Any], po2: PipeOutput[Any],
                **kwargs) -> PipeOutput:
        """
        This method is called to merge the output of two pipes.  It is called by the _run method when the pipe is a
        collate pipe.
        :param frm: From datetime
        :param to: To datetime
        :param args: any args passed to the get method
        :param po1: The output of the first pipeline (left)
        :param po2: The output of the second pipeline (right)
        :param kwargs: Any keyword args passed to the get method
        :return: A pipe output object that holds the output of the pipe and some metadata
        """
        raise NotImplementedError(f"Collate not implemented for object of type: {self.T}")

    def fetch(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput[Any]], **kwargs) -> PipeOutput:
        """
        This method is called to fetch the data from an external source.  It is called by the _run method when the pipe
        is a fetch pipe.
        :param frm: From datetime
        :param to: To datetime
        :param args: Any args passed to the get method
        :param po: The pipe output of the previous pipe
        :param kwargs: Any keyword args passed to the get method
        :return: A pipe output object that holds the output of the pipe and some metadata
        """
        raise NotImplementedError(f"Fetch not implemented for object of type: {self.T}")

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> PipeOutput:
        """
        This method is called to process and transform the data.  It is called by the _run method when the pipe is a
        process pipe.
        :param frm: From datetime
        :param to: To datetime
        :param args: Any args passed to the get method
        :param po: The pipe output of the previous pipe
        :param kwargs: Any keyword args passed to the get method
        :return: A pipe output object that holds the output of the pipe and some metadata
        """
        raise NotImplementedError(f"Process not implemented for object of type: {self.T}")

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput[Any], **kwargs) -> None:
        """
        This method is called to cache the output of the pipeline.  It is called by the _run method when the pipe is a
        cache pipe.
        :param frm: From datetime
        :param to: To datetime
        :param args: Any args passed to the get method
        :param po: The pipe output of the previous pipe
        :param kwargs: Any keyword args passed to the get method
        :return: None
        """
        raise NotImplementedError(f"Cache not implemented for object of type: {self.T}")

    def load(self) -> Optional[CacheObject]:
        """
        This method is called to load the cache from the disk.  It is called by the _load_cache method when the pipe is
        a cache pipe.  It will return the loaded cache object if found.  Otherwise, it will return None.
        :return: CacheObject if found else None
        """
        raise NotImplementedError(f"Load not implemented for object of type: {self.T}")

    def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
        """
        This method is called to revalidate the cache.  It is called by the _run method when the pipe is a cache pipe.
        :param frm: From datetime
        :param to: To datetime
        :param args: Any args passed to the get method
        :param po: The pipe output of the previous pipe
        :param kwargs: Any keyword args passed to the get method
        :return: A pipe output object that holds the output of the pipe and some metadata
        """
        raise NotImplementedError(f"Revalidate not implemented for object of type: {self.T}")

    def __str__(self):
        """
        This method is called when the print function is called on the pipe.  It returns a string representation of the
        pipeline.  (The pipeline is represented as a tree inside a box)
        """
        render = self.render()
        lines = render.split("\n")
        if lines[-1] == "":
            lines = lines[:-1]
        width = max(len(line) for line in lines)
        # Put the pipe in a bounding box
        top_line = f"┌ DataPipe({self.T}, {self.name}) "
        top_line = top_line.ljust(width + 3, "─") + "┐"
        width = len(top_line) - 4
        bottom_line = f"└{'─' * (len(top_line) - 2)}┘"
        lines = ["│ " + line.ljust(width) + " │" for line in [""] + lines + [""]]
        return "\n".join([top_line] + lines + [bottom_line])

    def __repr__(self):
        """
        This method is called when the repr function is called on the pipe.  It returns a short string representation of
        the pipe.
        """
        return f"DataPipe({self.T}, {self.name})"

    def __iter__(self) -> Iterator['DataPipe']:
        """
        To iter on each pipe in the pipeline.

        Warning:
            The pipes might not be in order.
        :return: An iterator on each pipe in the pipeline
        """
        pipes = []
        self._flatten(pipes)
        return iter(pipes)

    @property
    def pipe_id(self) -> int:
        """
        Get the pipe_id (Identifier unique to each pipe in the pipeline).  It is given at built time and is deterministic.
        This means that each time the pipeline is built, the pipe_id will be the same for each pipe, given that the
        pipeline structure is the same.
        :return: The pipe_id
        """
        return self._pipe_id

    def __len__(self) -> int:
        """
        Get the number of pipes in the pipeline.
        """
        pipes = []
        self._flatten(pipes)
        return len(pipes)


    def render(self):
        """
        This method is used to render the pipeline as a string.  It renders the pipe structure as a tree with each node
        being the name of the pipe.  It is used by the __str__ method to render the pipeline as a tree.
        :return: The rendered pipeline as a string
        """
        if self.T == DataPipeType.COLLATE and self._pipes is not None:
            pipe1 = self._pipes[0].render()
            pipe2 = self._pipes[1].render()
            n_lines1 = pipe1.count("\n")
            n_lines2 = pipe2.count("\n")
            # Easy exit
            if n_lines1 == 0 and n_lines2 == 0:
                pipe1 += " -> ┐\n"
                pipe2 += " -> ┘\n"
                if len(pipe1) > len(pipe2):
                    pipe2 = " " * (len(pipe1) - len(pipe2)) + pipe2
                elif len(pipe2) > len(pipe1):
                    pipe1 = " " * (len(pipe2) - len(pipe1)) + pipe1

                center = " " * (len(pipe1) - 2) + f"│ -> {self.name}\n"
                return pipe1 + center + pipe2

            if n_lines1 > 0:
                pipe1, longest_line1 = self._format_multiline_output(pipe1, "", join=False)
                pipe1[longest_line1] += "┐"
                pipe1 = pipe1[:-1]
                for i in range(longest_line1 + 1, len(pipe1)):
                    pipe1[i] += "│"
                # Fill with space the ones that doesn't have '|'
                pipe1 = [line.ljust(len(pipe1[-1])) for line in pipe1]
            else:
                pipe1 = [pipe1.rstrip("\n") + " -> ┐"]
            if n_lines2 > 0:
                pipe2, longest_line2 = self._format_multiline_output(pipe2, self._pipes[1].name, join=False)
                pipe2[longest_line2] += " -> ┘"
                for i in range(longest_line2):
                    pipe2[i] += "    │"
            else:
                pipe2 = [pipe2.rstrip("\n") + " -> ┘"]
            # Shift lines to the right
            line_len = max(len(pipe1[0]), len(pipe2[0]))
            pipe1 = [line.rjust(line_len) for line in pipe1]
            pipe2 = [line.rjust(line_len) for line in pipe2]
            center = " " * (line_len - 1) + f"│ -> {self.name}\n"
            return "\n".join(pipe1) + "\n" + center + "\n".join(pipe2) + "\n"
        else:
            if self._pipes is None:
                return f"{self.name}"
            else:
                prev_render = self._pipes.render()
                n_lines = prev_render.count("\n")
                if n_lines > 1:
                    return self._format_multiline_output(prev_render, self.name)
                else:
                    return prev_render + " -> " + self.name

    @staticmethod
    def _format_multiline_output(render: str, name: str, join: bool = True):
        prev_render = render.split("\n")
        prev_render_len = [len(line.rstrip(" ")) for line in prev_render]
        longest_line = prev_render_len.index(max(prev_render_len))
        prev_render[longest_line] = prev_render[longest_line].rstrip("\n") + " -> " + name
        good_len = len(prev_render[longest_line])
        prev_render = [line.ljust(good_len) for line in prev_render]
        if join:
            return "\n".join(prev_render[:-1]) + "\n"
        else:
            return prev_render, longest_line

if __name__ == "__main__":
    class Pipe(DataPipe):
        def __init__(self, T: DataPipeType, name: str = "", rev: bool = False):
            super().__init__(T)
            self.name = name
            self.rev = rev

        def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
            print(f"Fetching from {frm} to {to} Prev output: {po}")
            return PipeOutput([1,1,1,1,1], self)

        def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
            print(f"Processing from {frm} to {to} Prev output: {po}")
            return PipeOutput([2,2,2,2,2], self)

        def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
            print(f"Collating from {frm} to {to} Prev output: {po1} {po2}")
            return PipeOutput([v1 + v2 for v1, v2 in zip(po1.value, po2.value)], self)

        def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
            print(f"Caching from {frm} to {to} Prev output: {po}")
            self._cache = CacheObject(po, self._pipe_id)

        def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
            if self.rev:
                print(f"TRIGGERING FULL REVALIATION from {frm} to {to} Prev output: {po}")
                return RevalidateAction.FULL_REVALIDATE
            else:
                print(f"No revalidation from {frm} to {to} Prev output: {po}")
                return RevalidateAction.NOOP

    pipe1 = Pipe(DataPipeType.FETCH, "Fetch1") | Pipe(DataPipeType.PROCESS, "Process1") | Pipe(DataPipeType.CACHE, "Cache1", rev=True)
    pipe2 = Pipe(DataPipeType.FETCH, "Fetch2") | Pipe(DataPipeType.PROCESS, "process2") | Pipe(DataPipeType.CACHE, "Cache2")
    pipe3 = Pipe.Collate(pipe1, pipe2)
    pipe4 = pipe3 | Pipe(DataPipeType.PROCESS, "Process3")
    branch1 = Pipe(DataPipeType.FETCH, "Fetch4") | Pipe(DataPipeType.PROCESS, "Process4") | Pipe(DataPipeType.CACHE, "Cache4")
    print(Pipe.Collate(pipe4, branch1))
    print(pipe3.get(datetime(2023, 1, 1), datetime(2023, 1, 2)))
    print(pipe2._cache)
    print("-"*100)
    print(pipe3.get(datetime(2023, 1, 1), datetime(2023, 1, 2)))

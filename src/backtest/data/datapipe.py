from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any
from enum import Enum
from datetime import datetime
from copy import deepcopy

class DataPipeType(Enum):
    FETCH = "FETCH"
    PROCESS = "PROCESS"
    CACHE = "CACHE"
    COLLATE = "COLLATE"


class PipeOutput:
    def __init__(self, value: Any, output_from: 'DataPipe', revalidate_action: Optional['RevalidateAction'] = None,
                 **kwargs):
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
        self._revalidate_action = action

    @property
    def value(self):
        return self._value

    @property
    def revalidate(self):
        return self._revalidate_action

    def __str__(self):
        return f"PipeOutput({repr(self.value)}, from={self._output_from})"


class RevalidateAction(Enum):
    REVALIDATE = "REVALIDATE"
    FULL_REVALIDATE = "FULL_REVALIDATE"
    NOOP = "NOOP"

class DataPipe(ABC):
    def __init__(self, T: DataPipeType, name: Optional[str] = None):
        self._pipes: Optional[Union[DataPipe, List[DataPipe]]] = None
        self.T = T
        self._cache = None
        self.name = name if name is not None else self.__class__.__name__

    def get(self, frm: datetime, to: datetime, *args, **kwargs) -> Any:
        out = self._run(frm, to, *args, po=None, **kwargs)
        if out is None:
            raise ValueError(f"Pipe {self.name} returned None")
        # We run again the pipeline without using cache data.
        if out.revalidate == RevalidateAction.FULL_REVALIDATE:
            out = self._run(frm, to, *args, po=None, force_reload=True, **kwargs)

        # Unwrap the output (We do not want to return a PipeOutput object)
        return out.value

    def _run(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput] = None, force_reload: bool = False,
             **kwargs) -> PipeOutput:
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
            po = self.fetch(frm, to, *args, po=po, **kwargs)
            po.set_revalidate(rev)
        elif self.T == DataPipeType.PROCESS:
            rev = po.revalidate if po is not None else RevalidateAction.NOOP
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
                    po = PipeOutput(self._cache, output_from=self, revalidate_action=revalidate_action)
        else:
            raise NotImplementedError(f"DataPipeType {self.T} not implemented")
        return po

    def _build(self, other: 'DataPipe') -> 'DataPipe':
        # new = deepcopy(other)
        new = other
        new._pipes = self
        return new

    def __or__(self, other: 'DataPipe') -> 'DataPipe':
        return self._build(other)

    @classmethod
    def Collate(cls, pipe1: 'DataPipe', pipe2: 'DataPipe') -> 'DataPipe':
        new = cls(DataPipeType.COLLATE, name=f"Collate")
        new._pipes = [pipe1, pipe2]
        return new

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
        raise NotImplementedError("Collate not implemented")

    def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        raise NotImplementedError("Collate not implemented")

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        raise NotImplementedError("Collate not implemented")

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
        raise NotImplementedError("Collate not implemented")

    def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
        raise NotImplementedError("Collate not implemented")

    def __str__(self):
        render = self._render()
        lines = render.split("\n")
        width = max(len(line) for line in lines)
        # Put the pipe in a bounding box
        top_line = f"┌ DataPipe({self.T}, {self.name}) "
        top_line = top_line.ljust(width + 3, "─") + "┐"
        bottom_line = f"└{'─' * (len(top_line) - 2)}┘"
        lines = ["│ " + line.ljust(width) + " │" for line in [""] + lines + [""]]
        return "\n".join([top_line] + lines + [bottom_line])

    def __repr__(self):
        return f"DataPipe({self.T}, {self.name})"


    def _render(self):
        if self.T == DataPipeType.COLLATE:
            pipe1 = self._pipes[0]._render()
            pipe2 = self._pipes[1]._render()
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
            return "\n".join(pipe1) + "\n" + center + "\n".join(pipe2)
        else:
            if self._pipes is None:
                return f"{self.name}"
            else:
                prev_render = self._pipes._render()
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
            self._cache = po.value

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
    # print(pipe4)
    # print(pipe3.get(datetime(2023, 1, 1), datetime(2023, 1, 2)))
    # print(pipe3._cache)
    # print("-"*100)
    # print(pipe3.get(datetime(2023, 1, 1), datetime(2023, 1, 2)))

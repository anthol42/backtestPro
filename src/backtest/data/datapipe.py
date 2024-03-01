from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any
from enum import Enum
from datetime import datetime

class DataPipeType(Enum):
    FETCH = "FETCH"
    PROCESS = "PROCESS"
    CACHE = "CACHE"
    COLLATE = "COLLATE"
    ASSEMBLY = "ASSEMBLY"    # This is multiple pipes combined into one


class PipeOutput:
    def __init__(self):
        pass

    @property
    def value(self):
        pass


class RevalidateAction:
    pass

class DataPipe(ABC):
    def __init__(self, T: DataPipeType):
        self._pipes: List['DataPipe'] = []
        self.T = T
        self.cache: Any = None

    def get(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput] = None, **kwargs) -> Any:
        out: PipeOutput = None


        # Unwrap the output if this is the root of the graph (We do not want to return a PipeOutput object)
        if po is None:
            return out.value
        else:
            return po
    def run(self, frm: datetime, to: datetime, *args, po: Optional[PipeOutput] = None, **kwargs) -> PipeOutput:
        pass

    def _build(self, other: 'DataPipe') -> 'DataPipe':
        new = self.__class__(DataPipeType.ASSEMBLY)
        if len(self._pipes) == 0:
            new._pipes.append(self)
        else:
            new._pipes = self._pipes
        if len(other._pipes) == 0:
            new._pipes.append(other)
        else:
            new._pipes += other._pipes
        return new

    def __or__(self, other: 'DataPipe') -> 'DataPipe':
        return self._build(other)

    def collate(self, frm: datetime, to: datetime, *args, po1: PipeOutput, po2: PipeOutput, **kwargs) -> PipeOutput:
        pass

    def fetch(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        pass

    def process(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> PipeOutput:
        pass

    def cache(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> None:
        pass

    def revalidate(self, frm: datetime, to: datetime, *args, po: PipeOutput, **kwargs) -> RevalidateAction:
        pass
from .state_signals import StateSignals
from pathlib import PurePath
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union


class Renderer(ABC):
    def __init__(self):
        self._cb: Optional[Callable[[StateSignals, PurePath], None]] = None
        self.name: Optional[str] = None

    def __call__(self, cb: Callable[[StateSignals, PurePath], None]):
        self._cb = cb
        self.name = cb.__name__
        return self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Renderer({self.name})"

    def render(self, state: StateSignals, base_path: PurePath):
        """
        Renders the state and signals and save it to a file or multiple files inside the provided base path.
        :param state: The state and signals to render
        :param base_path: The base path to save the rendered files
        """
        if self._cb:
            self._cb(state, base_path)
        else:
            raise NotImplementedError("The render method must be implemented by the subclass")




# TODO: Implement a __str__ method similiar to the one in the IndicatorSet class
class RendererList:
    """
    Class to run multiple renderers at once
    """

    def __init__(self, *renderers: Renderer):
        self.renderers = renderers

    def render(self, state: StateSignals, base_path: PurePath):
        for renderer in self.renderers:
            renderer.render(state, base_path)
from .state_signals import StateSignals
from pathlib import PurePath
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
from copy import deepcopy


class Renderer(ABC):
    """
    This class is designed to convert a StateSignals object into a file or multiple files.  The goal of exporting a
    StateSignals object is to be able to visualize the data in a human-readable format or to export it into a
    program-readable format (example: JSON, CSV, etc) for further processing.
    The class can be used as a decorator or derived.  The derived class must implement the render method.

    Example:
        The first example is deriving the class and implementing the render method.
        >>> from backtest.serve import Renderer
        >>> class MyRenderer(Renderer):
        ...     def render(self, state: StateSignals, base_path: PurePath):
        ...         print(f"Rendering {state} to {base_path}")

        The second example is using the class as a decorator.
        >>> from backtest.serve import Renderer
        >>> @Renderer()
        ... def my_renderer(state: StateSignals, base_path: PurePath):
        ...     print(f"Rendering {state} to {base_path}")
    """
    def __init__(self):
        self._cb: Optional[Callable[[StateSignals, PurePath], None]] = None
        self.name: Optional[str] = self.__class__.__name__

    def __call__(self, cb: Optional[Callable[[StateSignals, PurePath], None]] = None) -> 'Renderer':
        new = deepcopy(self)
        if cb is not None:
            new._cb = cb
            new.name = cb.__name__
        return new

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




class RendererList:
    """
    A class that have the same structure as a Renderer, but it contains a list of renderers.  This class is used to
    combine multiple renderers into a single renderer.  When the render method is called, it will call the render method
    of each renderer in the list.

    Note: The order of the renderers in the list is kept.
    """

    def __init__(self, *renderers: Renderer):
        self.renderers = renderers

    def render(self, state: StateSignals, base_path: PurePath):
        """
        Run multiple renderers with the given state and base path.
        :param state: The state and signals to render
        :param base_path: The root path to store the rendered files
        :return: None
        """
        for renderer in self.renderers:
            renderer.render(state, base_path)


    def __str__(self):
        s = "RendererList(\n"
        for r in self.renderers:
            s += f"    {r}\n"
        s += ")"
        return s

    def __repr__(self):
        return f"RendererList(len={len(self.renderers)})"
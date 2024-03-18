from pathlib import PurePath

from .markup_renderer import MarkupObject, MarkupRenderer
from ..state_signals import StateSignals


class HTMLRenderer(MarkupRenderer):
    """
    Class designed to render a python object to a markup language.  This can be xml, html, markdown, etc.
    """
    def __init__(self):
        super().__init__()

    def render(self, state: StateSignals, base_path: PurePath):
        pass
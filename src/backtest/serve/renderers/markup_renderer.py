from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set
from ..renderer import Renderer
import re

class MarkupObject:
    def __init__(self, template: str, format_: str = "<.*?>"):
        """

        :param template:
        :param format: This is a regex expression that should contain only one '.*?'.  It corresponds to the key
        :param trailing_format: his is a regex expression that should contain only one '.*?'.  It corresponds to the key
        """
        self.template = template
        self.format_ = format_
        self.keys = self.extract_keys(template, self.format_)


    @staticmethod
    def extract_keys(template: str, format_: str) -> Set[str]:
        """
        Extracts the keys from the template
        """
        # Leading
        matchs = re.findall(format_, template)
        group_pattern = format_.replace(".*?", "(.*?)")
        keys = {re.search(group_pattern, m).group(1) for m in matchs}

        return keys


    def render(self, data: Dict[str, str]) -> str:
        """
        Renders the template with the data
        """
        out = self.template
        for key in self.keys:
            tag = self.format_.replace(".*?", key)
            out = out.replace(tag, data[key])
        return out

    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f"MarkupObject(format_={self.format_})"





class MarkupRenderer(Renderer):
    """
    Class designed to render a python object to a markup language.  This can be xml, html, markdown, etc.
    """
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    template = (
        "<html>"
        "<head>"
        "<title>{TITLE}<title>"
        "</head>"
        "<body>"
        "<h1>{TITLE}<h1>"
        "<p>{DESCRIPTION}<p>"
        "</body>"
        "</html>")
    markup = MarkupObject(template, format_="{.*?}")
    print(markup.render({"TITLE": "Hello World", "DESCRIPTION": "This is a test"}))
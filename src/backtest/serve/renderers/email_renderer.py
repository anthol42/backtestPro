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
from pathlib import PurePath
from .markup_renderer import MarkupObject, MarkupRenderer, format_number
from ..state_signals import StateSignals, ServerStatus
from typing import Optional
import os
import shutil
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


TEMPLATE_PATH = PurePath(f"{os.path.dirname(__file__)}/html_templates")

def format_lim(a: Optional[float]) -> str:
    """
    Replace a with N/A if a is None.  Else, it returns the float with a $ sign.
    :param a: Float
    :return: str
    """
    if a is None:
        return "N/A"
    return f"{a:,}".replace(",", " ") + "$"

class EmailRenderer(MarkupRenderer):
    """
    It renders only the trading signals in a html file.  As opposed to the HTMLRenderer which renders the performances,
    the portfolio and other useful stats, this renderer is more lightweight and is designed to be sent by email.
    This being said, it is designed to be viewed on mobile devices.  It also looks good on desktop though.
    The three styles are:
    - light: A light theme
    - dark: A dark theme (Darcula style)
    - rich: Another dark theme with another color palette
    """
    def __init__(self, style: str = "light", filename: str = "email_report.html"):
        super().__init__()
        styles = ["light", "dark", "rich"]
        if style not in styles:
            raise ValueError(f"Style {style} does not exists.  Available styles are {styles}")
        self.style = style
        self.filename = filename

    def render(self, state: StateSignals, base_path: PurePath):

        card_template = self.components["SIGNAL_CARD"]
        cards = []
        for signal in state:
            if signal.timestamp == state.timestamp:
                # New signal
                cards.append(card_template.render(dict(new='<span class="signal-badge">New</span>', dt=signal.timestamp,
                                                       ticker=signal.security, amount=signal.amount,
                                                       amount_borrowed=signal.amount_borrowed,
                                                       price_l=format_lim(signal.security_price_limit[0]),
                                                       price_h=format_lim(signal.security_price_limit[1]),
                                                       expiry=signal.expiry)))
            else:
                cards.append(card_template.render(dict(new='', dt=signal.timestamp, ticker=signal.security,
                                                       amount=signal.amount, amount_borrowed=signal.amount_borrowed,
                                                       price_l=format_lim(signal.security_price_limit[0]),
                                                       price_h=format_lim(signal.security_price_limit[1]),
                                                       expiry=signal.expiry)))
        self.prerender("SCROLL_SIGNALS", cards="\n".join(cards))

        if state.status == ServerStatus.ERROR:
            color = "red"
        elif state.status == ServerStatus.WARNING:
            color = "orange"
        else:
            color = "green"

        # Handle css
        css = self.get_style(self.style)

        # Render the whole
        html_content = self.render_template(status_color=color, status=state.status.name, title="Financial Report",
                                            style=css, script="", containerS="", containerE="")

        # Save the file
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(base_path / self.filename, "w") as f:
            f.write(html_content)
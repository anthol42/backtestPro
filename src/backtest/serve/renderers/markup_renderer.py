from typing import Any, Dict, List, Optional, Tuple, Set
from ..renderer import Renderer
import re
from pathlib import PurePath
from datetime import datetime
from ..state_signals import StateSignals, ServerStatus
from ..stats_calculator import StatCalculator
from ...engine import TradeOrder, Position, TSData
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import os
import shutil
import glob

try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


def format_number(number: Optional[float], dec: int = 2) -> str:
    """
    Formats a number to a string with a fixed number of decimals.  It also put spaces every 3 digits for big numbers.
    Example:
        11235.4242 -> "11 235.42"
    """
    if number is None:
        return "N/A"
    if number < 1000:
        return f"{round(number, dec)}"
    else:
        return f"{round(number, dec):,}".replace(",", " ")

class MarkupObject:
    def __init__(self, template: str, format_: str = "{{.*?}}"):
        """

        :param template:
        :param format: This is a regex expression that should contain only one '.*?'.  It corresponds to the key
        :param trailing_format: his is a regex expression that should contain only one '.*?'.  It corresponds to the key
        """
        self.template = template
        self.format_ = format_
        self.keys = self.extract_keys(template, self.format_)
        self.data: Optional[Dict[str, str]] = None


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

    def __call__(self, data: Optional[Dict[str, str]] = None, **kwargs: str) -> 'MarkupObject':
        """
        Save the data internally without rendering the object.  Kind of a pre-render.
        :param data: The data to render the template
        :param kwargs: It is also possible to pass the data as kwargs
        :return: self
        """
        if data is None:
            data = kwargs
        for key in self.keys:
            if key not in data:
                raise RuntimeError(f"Key {key} is defined in the template, but not specified in the data")
        self.data = data
        return self

    def render(self, data: Optional[Dict[str, str]] = None, **kwargs) -> str:
        if data is None:
            data = kwargs
        if len(data) == 0 and self.data is not None:
            data = self.data
        out = self.template
        for key in self.keys:
            tag = self.format_.replace(".*?", key)
            if key not in data:
                raise RuntimeError(f"Key {key} is defined in the template, but not specified in the data")
            out = out.replace(tag, str(data[key]))
        return out

    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f"MarkupObject(format_={self.format_})"





class MarkupRenderer(Renderer):
    """
    Class designed to facilitate the creation of custom reports renderer.  It includes a set of method that helps
    extract the relevant data from the state object and render in a hierarchical manner.
    """
    def __init__(self, template_path: PurePath = PurePath(f"{os.path.dirname(__file__)}/html_templates")):
        super().__init__()
        self.template_path = template_path

        # Load the template and the components
        # 1: load the main file
        with open(self.template_path / "main.html", "r") as f:
            self.template = MarkupObject(f.read())

        # 2: Load the components
        self.components: Dict[str, MarkupObject] = {}
        components_paths = glob.glob(str(self.template_path / "components" / "*.html"))
        for path in components_paths:
            with open(path, "r") as f:
                self.components[PurePath(path).name.split(".")[0]] = MarkupObject(f.read())
        self.rendered_components: Dict[str, str] = {}

        # Index stylesheets
        self.stylesheets = {}
        for path in glob.glob(str(self.template_path / "*.css")) + glob.glob(str(self.template_path / "styles" / "*.css")):
            self.stylesheets[PurePath(path).name.split(".")[0]] = path

        # Index scripts
        self.scripts = {}
        for path in glob.glob(str(self.template_path / "*.js")) + glob.glob(str(self.template_path / "scripts" / "*.js")):
            self.scripts[PurePath(path).name.split(".")[0]] = path

    def get_style(self, name: str) -> str:
        """
        Load any stylesheet that is in the 'template_path' main directory, or in a styles subdirectory.
        :param name: The name of the file without the .css.  It should be in the main directory or in the styles subdirectory.
        :return: The text in the file.
        """
        if name not in self.stylesheets:
            raise ValueError(f"Stylesheet {name} does not exists.  Available stylesheets are {list(self.stylesheets.keys())}")
        with open(self.stylesheets[name], "r") as f:
            return f.read()

    def get_script(self, name: str) -> str:
        """
        Load any script that is in the 'template_path' main directory, or in a scripts subdirectory.
        :param name: The name of the file without the .js.  It should be in the main directory or in the scripts subdirectory.
        :return: The text in the file.
        """
        if name not in self.scripts:
            raise ValueError(f"Script {name} does not exists.  Available scripts are {list(self.scripts.keys())}")
        with open(self.scripts[name], "r") as f:
            return f.read()



    def prerender(self, component: str, **kwargs) -> str:
        """
        Render the component with the given data.  It stores internally the rendered component.
        By default, if a component depends on another component that hasn't been rendered, it is assumed that the
        component was optional and it will be rendered as an empty string (ignored).
        :param component: The component name
        :param kwargs: The data to render the component
        :return: The rendered component (text)
        """
        if component not in self.components:
            raise ValueError(f"Component {component} does not exists.  Available components are {list(self.components.keys())}")
        data = kwargs
        for key in self.components[component].keys:
            if key not in data:
                if key in self.rendered_components:
                    data[key] = self.rendered_components[key]
                elif key in self.components:
                    data[key] = ""
                else:
                    raise RuntimeError(f"Key {key} is defined in the component, but not specified in the data nor in the components.")
        self.rendered_components[component] = self.components[component].render(kwargs)
        return self.rendered_components[component]


    def render_template(self, **kwargs) -> str:
        """
        Render the main template with the given data.  It will automatically map pre-rendered components to their
        corresponding tag in the template.
        By default, if a component is required to render the main template, but hasn't been rendered, the component will
        be rendered as an empty string (ignored).
        :param kwargs: The data to add that isn't in the components.
        :return: The rendered template as a string.
        """
        keys = self.template.keys
        data = kwargs
        for key in keys:
            if key not in data:
                if key in self.rendered_components:
                    data[key] = self.rendered_components[key]
                elif key in self.components:
                    data[key] = ""
                else:
                    raise RuntimeError(f"Key {key} is defined in the template, but not specified in the data nor in the components.")
        return self.template.render(data)

    def get_performance_data(self, state: StateSignals,
                             cutoff_delta: pd.Timedelta = pd.Timedelta(days=365)) -> Tuple[pd.Series, pd.Series, str]:
        """
        Build a series of the portfolio worth and the index worth for easier comparison.
        :param state: The StateSignals object
        :param cutoff_delta: The amount of time before which the data is not considered.
        :return: The historical worth of the portfolio for the given period (cutoff_days) and the historical worth of the
        index for the same period, the name of the index took.  (In case there are more than 1, only one is used)
        """
        cutoff = state.timestamp - cutoff_delta
        portfolio_worth = np.array([s.worth for s in state.broker.historical_states if s.timestamp > cutoff])
        security_names, data = self.prepare_data(state)
        current_worth = state.broker.get_worth(security_names, data)
        portfolio_worth = np.append(portfolio_worth, current_worth)
        idx_name = list(state.index_data.keys())[0]
        index_worth = state.index_data[idx_name].data["Close"].loc[cutoff:state.timestamp].values
        index_timestamps = state.index_data[idx_name].data["Close"].loc[cutoff:state.timestamp].index
        portfolio_timestamps = pd.DatetimeIndex([s.timestamp for s in state.broker.historical_states if s.timestamp > cutoff])
        portfolio_timestamps = portfolio_timestamps.append(pd.DatetimeIndex([state.timestamp]))
        worth = pd.Series(portfolio_worth, index=portfolio_timestamps)
        index = pd.Series(index_worth, index=index_timestamps)
        return worth, index, idx_name

    @staticmethod
    def prepare_data(state: StateSignals) -> (List[str], np.ndarray):
        """
        Extracts the relevant up-to-date data (current timestep) from the state object and returns it as an array with
        the corresponding security names.
        """
        data = state.data[state.main_idx]
        security_names = list(data.keys())
        out = np.empty((len(security_names), 4), dtype=np.float32)    # Shape(n, 4) for OHLC
        for i, name in enumerate(security_names):
            out[i, 0] = data[name].data["Open"].iloc[-1]
            out[i, 1] = data[name].data["High"].iloc[-1]
            out[i, 2] = data[name].data["Low"].iloc[-1]
            out[i, 3] = data[name].data["Close"].iloc[-1]

        return security_names, out


    @staticmethod
    def get_ticker_info(ticker: str, data: List[Dict[str, TSData]], main_idx: int) -> Dict[str, float]:
        """
        :return The change in dollars, the change in percent and its current value
        """
        if data is None:
            raise ValueError("No data was provided.  The html renderer needs the data to render the portfolio.")
        if main_idx is None:
            raise ValueError("No main index was provided.  The html renderer needs the main index to render the portfolio.")

        # Get ticker data
        ticker_data = data[main_idx][ticker].data

        # Get the change in dollars
        change_dollars = ticker_data["Close"].iloc[-1] - ticker_data["Close"].iloc[-2]
        change_percent = 100 * change_dollars / ticker_data["Close"].iloc[-2]

        return {
            "change_dollars": change_dollars,
            "change_percent": change_percent,
            "current_value": ticker_data["Close"].iloc[-1]
        }



    @staticmethod
    def render_portfolio(position: Position, ticker_info: Dict[str, float]) -> str:
        """
        Renders each position as a html row [str]
        """
        # Columns: Ticker, Average_buy_price, Quantity, Change ($), Change (%), Unrealized P&L ($), Unrealized P&L (%)
        template = ("<tr><td>{{ticker}}</td><td>{{average_buy_price}}</td><td>{{quantity}}</td><td>{{change_dollars}}</td>"
                    "<td>{{change_percent}}</td><td>{{unrealized_pl_dollars}}</td><td>{{unrealized_pl_percent}}</td></tr>")
        format_ = "{{.*?}}"
        markup = MarkupObject(template, format_)
        gains = position.amount * ticker_info["current_value"] - position.amount * position.average_price
        data = {"ticker": position.ticker,
                "average_buy_price": format_number(position.average_price, 2),
                "quantity": position.amount,
                "change_dollars": format_number(ticker_info["change_dollars"], 2),
                "change_percent": format_number(ticker_info["change_percent"], 2),
                "unrealized_pl_dollars": format_number(gains, 2),
                "unrealized_pl_percent": format_number(100 * gains / (position.amount * position.average_price), 2)
                }
        return markup.render(data)

    @staticmethod
    def render_signal(signal: TradeOrder):
        """
        Renders each signal as a html row [str]
        """
        template = ("<tr><td>{{timestamp}}</td><td>{{ticker}}</td><td>{{signal_type}}</td><td>{{price_limitl}}</td>"
                    "<td>{{price_limith}}</td><td>{{quantity}}</td><td>{{qty_borrowed}}</td><td>{{expiry}}</td></tr>")
        format_ = "{{.*?}}"
        markup = MarkupObject(template, format_)
        data = {"timestamp": signal.timestamp.isoformat(),
                "ticker": signal.security,
                "signal_type": signal.trade_type.value,
                "price_limitl": signal.security_price_limit[0],
                "price_limith": signal.security_price_limit[1],
                "quantity": signal.amount,
                "qty_borrowed": signal.amount_borrowed,
                "expiry": signal.expiry.isoformat() if signal.expiry is not None else "None"}

        return markup.render(data)

    @staticmethod
    def chart_builder(portfolio_worth: np.ndarray, porfolio_timestamps: pd.DatetimeIndex, index_worth: np.ndarray,
                      index_timestamps: pd.DatetimeIndex, dark_theme: bool = False, index_name: str = "Index"):
        if not PLOTLY_INSTALLED:
            raise ImportError("Plotly is not installed.  Please install it to use the HTML renderer.")

        if len(portfolio_worth) == 0:
            return go.Figure()
        # Calculate percentage change
        portfolio_percentage = (portfolio_worth - portfolio_worth[0]) / portfolio_worth[0] * 100
        index_percentage = (index_worth - index_worth[0]) / index_worth[0] * 100

        # Create figure
        fig = go.Figure()

        # Add portfolio data
        color = 'blue' if not dark_theme else 'royalblue'
        fig.add_trace(go.Scatter(x=porfolio_timestamps, y=portfolio_percentage, name='Portfolio',
                                 line=dict(color=color, width=2)))

        # Add index data
        color = 'green' if not dark_theme else 'limegreen'
        fig.add_trace(go.Scatter(x=index_timestamps, y=index_percentage, name=index_name,
                                 line=dict(color=color, width=2, dash='dash')))

        # Update layout
        text_color = 'black' if not dark_theme else 'white'
        grid_color = 'lightgray' if not dark_theme else 'darkgray'
        fig.update_layout(title='Portfolio Performance',
                          xaxis_title='Date',
                          yaxis_title='Percentage Change',
                          legend=dict(x=0.02, y=0.98),
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Arial, sans-serif', size=12, color=text_color),
                          yaxis=dict(showgrid=True, gridwidth=1, gridcolor=grid_color, tickformat=".1f"))

        return fig
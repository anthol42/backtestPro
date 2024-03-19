from pathlib import PurePath
from datetime import datetime
from .markup_renderer import MarkupObject, MarkupRenderer
from ..state_signals import StateSignals
from ...engine import TradeOrder, Position, TSData, BackTestResult
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import os
import shutil
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


TEMPLATE_PATH = PurePath(f"{os.path.dirname(__file__)}/html_templates")

# TODO: Create a Stats calculator like the backtest results, but with a moving window of 1 year.

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

class HTMLRenderer(MarkupRenderer):
    """
    Class designed to render a python object to a markup language.  This can be xml, html, markdown, etc.
    It expects that the tradeSignals object contains the index data, the tickers data and the main idx of the tickers
    data.
    """
    def __init__(self, style: str = "light"):
        super().__init__()
        styles = ["light", "dark", "rich"]
        if style not in styles:
            raise ValueError(f"Style {style} does not exists.  Available styles are {styles}")
        self.style = style

    def render(self, state: StateSignals, base_path: PurePath):
        # Step 0: Load the template
        with open(TEMPLATE_PATH / "main.html", "r") as f:
            template = f.read()

        # Step 1: Render signals
        signal_list = "\n".join(self.render_signal(signal) for signal in state)

        # Step 2: Render The portfolio
        long_portfolio = "\n".join(
            self.render_portfolio(position, self.get_ticker_info(position.ticker, state.data, state.main_idx))
                                   for _, position in state.portfolio.getLong().items() if position.amount > 0
        )

        # Step 3: Render the short portfolio
        short_portfolio = "\n".join(
            self.render_portfolio(position, self.get_ticker_info(position.ticker, state.data, state.main_idx))
                                     for _, position in state.portfolio.getShort().items() if position.amount > 0
        )

        # Step 4: Render the chart performances
        # Get the portfolio worth from one year ago to now
        cutoff = state.timestamp - pd.Timedelta(days=365)
        portfolio_worth = np.array([s.worth for s in state.broker.historical_states if s.timestamp > cutoff])
        security_names, data = self.prepare_data(state)
        current_worth = state.broker.get_worth(security_names, data)
        portfolio_worth = np.append(portfolio_worth, current_worth)
        idx_name = list(state.index_data.keys())[0]
        index_worth = state.index_data[idx_name].data["Close"].loc[cutoff:state.timestamp].values
        index_timestamps = state.index_data[idx_name].data["Close"].loc[cutoff:state.timestamp].index
        index_name = state.index_data[idx_name].name
        portfolio_timestamps = pd.DatetimeIndex([s.timestamp for s in state.broker.historical_states if s.timestamp > cutoff])
        portfolio_timestamps = portfolio_timestamps.append(pd.DatetimeIndex([state.timestamp]))
        isDark = self.style == "dark" or self.style == "rich"
        fig = self.chart_builder(portfolio_worth, portfolio_timestamps, index_worth, index_timestamps, isDark, index_name)
        chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Step 5: Get the statistics
        if len(portfolio_worth) > 1:    # TODO: Change to 0 when using the new stats calculator
            res = BackTestResult("", None, portfolio_timestamps[0],portfolio_timestamps[-1], state.initial_cash,
                                 state.cash_controller._total_deposited, index_worth, state.broker, state.account)

            # Step 6: Bank account
            available_cash = state.account.get_cash()
            net_worth = state.broker.historical_states[-1].worth
            total_deposits = state.cash_controller._total_deposited
            collateral = state.account.collateral

            # Step 8: Render the template
            markup = MarkupObject(template, "{{.*?}}")
            data = {
                # Signals
                "signals": signal_list,
                # Portfolio
                "long_positions": long_portfolio,
                "short_positions": short_portfolio,
                # Chart
                "chart": chart,
                # Statistics
                "all_time_returns": format_number(res.returns),
                "annual_returns": format_number(res.annual_returns),
                "avg_drawdown": format_number(res.avg_drawdown),
                "avg_trade": format_number(res.avg_trade),
                "avg_trade_dur": format_number(res.avg_trade_duration),
                "best_trade": format_number(res.best_trade),
                "calmar_ratio": format_number(res.calmar_ratio),
                "current_equity": format_number(net_worth),
                "equity_peak": format_number(res.equity_peak),
                "index_returns": format_number(res.index_returns),
                "max_drawdown": format_number(res.max_drawdown),
                "max_trade_dur": format_number(res.max_trade_duration),
                "min_trade_dur": format_number(res.min_trade_duration),
                "profit_factor": format_number(res.profit_factor),
                "sharp_ratio": format_number(res.sharp_ratio),
                "sortino_ratio": format_number(res.sortino_ratio),
                "sqn": format_number(res.sqn),
                "win_rate": format_number(res.win_rate),
                "worst_trade": format_number(res.worst_trade),
                "yearly_exits": format_number(res.num_exits),
                "yearly_trades": format_number(res.num_trades),
                # Account
                "net_worth": format_number(net_worth),
                "deposits": format_number(total_deposits),
                "collateral": format_number(collateral),
                "avail_cash": format_number(available_cash)
            }
        else:
            markup = MarkupObject(template, "{{.*?}}")
            data = {
                # Signals
                "signals": signal_list,
                # Portfolio
                "long_positions": long_portfolio,
                "short_positions": short_portfolio,
                # Chart
                "chart": "NO DATA TO PLOT",
                # Statistics
                "all_time_returns": 0,
                "annual_returns": 0,
                "avg_drawdown": 0,
                "avg_trade": 0,
                "avg_trade_dur": 0,
                "best_trade": 0,
                "calmar_ratio": 0,
                "current_equity": 0,
                "equity_peak": 0,
                "index_returns": 0,
                "max_drawdown": 0,
                "max_trade_dur": 0,
                "min_trade_dur": 0,
                "profit_factor": 0,
                "sharp_ratio": 0,
                "sortino_ratio": 0,
                "sqn": 0,
                "win_rate": 0,
                "worst_trade": 0,
                "yearly_exits": 0,
                "yearly_trades": 0,
                # Account
                "net_worth": format_number(state.account.get_cash()),
                "deposits": format_number(state.cash_controller._total_deposited),
                "collateral": 0,
                "avail_cash": format_number(state.account.get_cash())
            }
        html_content = markup.render(data)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(base_path / "index.html", "w") as f:
            f.write(html_content)

        # Handle css
        if self.style == "dark":
            shutil.copy(TEMPLATE_PATH / "dark.css", base_path / "style.css")
        elif self.style == "rich":
            shutil.copy(TEMPLATE_PATH / "rich.css", base_path / "style.css")
        else:
            shutil.copy(TEMPLATE_PATH / "light.css", base_path / "style.css")


    @staticmethod
    def prepare_data(state: StateSignals) -> (List[str], np.ndarray):
        """
        Extracts the relevent up-to-date data from the state object and returns it as an array.
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
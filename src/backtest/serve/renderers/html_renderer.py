from pathlib import PurePath
from .markup_renderer import MarkupObject, MarkupRenderer, format_number
from ..state_signals import StateSignals, ServerStatus
from ..stats_calculator import StatCalculator
import os
import shutil
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


TEMPLATE_PATH = PurePath(f"{os.path.dirname(__file__)}/html_templates")

class HTMLRenderer(MarkupRenderer):
    """
    Class designed to render a python object to a markup language.  This can be xml, html, markdown, etc.
    It expects that the tradeSignals object contains the index data, the tickers data and the main idx of the tickers
    data.
    Stats are calculated based on a year of data except when specified all-time.  In that case, the stats are calculated
    from the first data point to the last.
    """
    def __init__(self, style: str = "light"):
        super().__init__()
        styles = ["light", "dark", "rich"]
        if style not in styles:
            raise ValueError(f"Style {style} does not exists.  Available styles are {styles}")
        self.style = style

    def render(self, state: StateSignals, base_path: PurePath):
        # Step 1: Render signals
        self.prerender("SIGNALS", signals="\n".join(self.render_signal(signal) for signal in state))

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
        self.prerender("PORTFOLIO", long=long_portfolio, short=short_portfolio)

        # Step 4: Render the chart performances
        # Get the portfolio worth from one year ago to now
        portfolio_worth, index_worth, index_name = self.get_performance_data(state)
        isDark = self.style == "dark" or self.style == "rich"
        fig = self.chart_builder(portfolio_worth.values, portfolio_worth.index, index_worth.values,
                                 index_worth.index, isDark, index_name)
        chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Step 5: Get the statistics
        if len(portfolio_worth) > 0:
            res = StatCalculator(state)

            # Step 6: Bank account
            available_cash = state.account.get_cash()
            net_worth = portfolio_worth.values[-1]
            monthly_deposits = state.cash_controller.monthly_variation(state.timestamp)
            collateral = state.account.collateral
            self.prerender("CASH", avail_cash=format_number(available_cash),
                           net_worth=format_number(net_worth), deposits=format_number(monthly_deposits),
                            collateral=format_number(collateral))

            # Step 7: Render the stats
            self.prerender("PERFORMANCES", chart=chart,
                            all_time_returns=format_number(res.all_time_returns),
                            year_returns=format_number(res.year_returns),
                            annual_returns=format_number(res.annual_returns),
                            current_equity=format_number(net_worth),
                            avg_drawdown=format_number(res.avg_drawdown),
                            avg_trade=format_number(res.avg_trade),
                            avg_trade_dur=format_number(res.avg_trade_duration),
                            best_trade=format_number(res.best_trade),
                            calmar_ratio=format_number(res.calmar_ratio),
                            equity_peak=format_number(res.equity_peak),
                            index_returns=format_number(res.index_returns),
                            max_drawdown=format_number(res.max_drawdown),
                            max_trade_dur=format_number(res.max_trade_duration),
                            min_trade_dur=format_number(res.min_trade_duration),
                            profit_factor=format_number(res.profit_factor),
                            sharp_ratio=format_number(res.sharp_ratio),
                            sortino_ratio=format_number(res.sortino_ratio),
                            sqn=format_number(res.sqn),
                            win_rate=format_number(res.win_rate),
                            worst_trade=format_number(res.worst_trade),
                            yearly_exits=format_number(res.num_exits),
                            yearly_trades=format_number(res.num_trades))
        else:
            self.prerender("CASH", avail_cash=format_number(state.account.get_cash()),
                           net_worth=format_number(state.account.get_cash()),
                           deposits=format_number(state.cash_controller._total_deposited), collateral=0)
            self.prerender("PERFORMANCES", chart="NO DATA TO PLOT",
                           all_time_returns=0,
                           year_returns=0,
                           annual_returns=0,
                           current_equity=0,
                           avg_drawdown=0,
                           avg_trade=0,
                           avg_trade_dur=0,
                           best_trade=0,
                           calmar_ratio=0,
                           equity_peak=0,
                           index_returns=0,
                           max_drawdown=0,
                           max_trade_dur=0,
                           min_trade_dur=0,
                           profit_factor=0,
                           sharp_ratio=0,
                           sortino_ratio=0,
                           sqn=0,
                           win_rate=0,
                           worst_trade=0,
                           yearly_exits=0,
                           yearly_trades=0)
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
                                            style=css, script="")

        # Save the file
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(base_path / "index.html", "w") as f:
            f.write(html_content)
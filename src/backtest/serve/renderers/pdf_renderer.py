from .markup_renderer import MarkupObject, MarkupRenderer, format_number
from pathlib import PurePath
from typing import List, Dict, Any, Optional
from ...engine import Position
import os
from ..state_signals import StateSignals, ServerStatus
from ..stats_calculator import StatCalculator
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_INSTALLED = True
    import logging
except ImportError:
    WEASYPRINT_INSTALLED = False
import os

TEMPLATE_PATH = PurePath(f"{os.path.dirname(__file__)}/pdf_templates")
ROW_PER_PAGE = 20

class PDFPage(MarkupObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: List[MarkupObject] = []

    def append(self, obj: MarkupObject):
        """
        Append a child object to the page.  The child objects must be pre-rendered before calling this method.
        Call the object and gives it the necessary data to render before appending it to the page.
        :param obj: The object to append to the page
        :return: None
        """
        self.children.append(obj)

    def render(self, *args, **kwargs) -> str:
        return "\n".join(child.render() for child in self.children)



class PDFDoc(MarkupObject):
    def __init__(self, header: Optional[MarkupObject], footer: Optional[MarkupObject], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []
        self.header = header
        self.footer = footer

    def new_page(self, header: bool = False, footer: bool = False) -> PDFPage:
        """
        Creates a new page and returns a reference to it.
        :return: The new page object
        """
        page = PDFPage("")
        if header and self.header is not None:
            page.append(self.header)
        if footer and self.footer is not None:
            page.append(self.footer)
        self.pages.append(page)
        return page

    def render(self, data: Optional[Dict[str, str]] = None, **kwargs) -> str:
        if data is None:
            data = kwargs
        if data is None:
            data = self.data
        out = self.template
        data["body"] = '<div style="break-before: page;"></div>'.join(page.render() for page in self.pages)
        for key in self.keys:
            tag = self.format_.replace(".*?", key)
            if key not in data:
                raise RuntimeError(f"Key {key} is defined in the template, but not specified in the data")
            out = out.replace(tag, str(data[key]))
        return out



class PDFRenderer(MarkupRenderer):
    def __init__(self, report_title: str = "Financial Report", style: str = "light"):
        super().__init__(TEMPLATE_PATH)
        styles = ["light", "dark", "rich"]
        if style not in styles:
            raise ValueError(f"Style {style} does not exists.  Available styles are {styles}")
        self.style = style
        self.report_title = report_title

    def render(self, state: StateSignals, base_path: PurePath):
        if not WEASYPRINT_INSTALLED:
            raise ImportError("Weasyprint is not installed.  Please install it to use this renderer")
        header = self.components["HEADER"](title=self.report_title)
        if state.status == ServerStatus.ERROR:
            color = "red"
        elif state.status == ServerStatus.WARNING:
            color = "orange"
        else:
            color = "green"

        footer = self.components["FOOTER"](status_color=color, status=state.status.name)
        pdf = PDFDoc(header, footer, self.template.template)

        # To save the chart
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # Step 1: Render signals
        signals = self.components["SIGNALS"](signals="\n".join(self.render_signal(signal) for signal in state))
        page = pdf.new_page(header=True)
        page.append(signals)

        # Step 2: Render The portfolio
        page = pdf.new_page(header=True)
        page.append(self.components["PORTFOLIO_HEADER"]())    # This opens the section, but we need to close it.
        long = [
            self.render_portfolio(position, self.get_ticker_info(position.ticker, state.data, state.main_idx))
                                   for _, position in state.portfolio.getLong().items() if position.amount > 0]
        long_tables = [long[i:i + ROW_PER_PAGE] for i in range(0, len(long), ROW_PER_PAGE)]
        page.append(self.components["PORTFOLIO"](table_name="Long Portfolio"))
        if len(long_tables) == 0:
            # Empty table
            page.append(self.components["PORTFOLIO_TABLE"](rows=""))
        else:
            for i, rows in enumerate(long_tables):
                table = self.components["PORTFOLIO_TABLE"](rows="\n".join(rows))
                page.append(table)
                page = pdf.new_page()

        # Step 3: Render the short portfolio
        short = [
            self.render_portfolio(position, self.get_ticker_info(position.ticker, state.data, state.main_idx))
                                     for _, position in state.portfolio.getShort().items() if position.amount > 0]
        short_tables = [short[i:i + ROW_PER_PAGE] for i in range(0, len(short), ROW_PER_PAGE)]
        page.append(self.components["PORTFOLIO"](table_name="Short Portfolio", name="short"))
        if len(short_tables) == 0:
            # Empty table
            page.append(self.components["PORTFOLIO_TABLE"](rows=""))
        else:
            for i, rows in enumerate(short_tables):
                table = self.components["PORTFOLIO_TABLE"](rows="\n".join(rows))
                page.append(table)
                if i == len(short_tables) - 1:
                    break    # We do not add a new page for the last one
                else:
                    page = pdf.new_page()

        page.append(MarkupObject("    </section>"))  # Close the section

        # Step 4: Render the chart performances
        # Get the portfolio worth from one year ago to now
        portfolio_worth, index_worth, index_name = self.get_performance_data(state)
        isDark = self.style == "dark" or self.style == "rich"
        fig = self.chart_builder(portfolio_worth.values, portfolio_worth.index, index_worth.values,
                                 index_worth.index, isDark, index_name)

        chart_path = base_path / "chart.svg"
        with open(base_path / "chart.svg", "w") as f:
            f.write(fig.to_image(format="svg").decode())
        page = pdf.new_page(header=True)
        page.append(self.components["PERFORMANCES"](chart_path=f"file://{PurePath(os.getcwd()) / chart_path}"))

        # Step 5: Get the statistics
        if len(portfolio_worth) > 0:
            res = StatCalculator(state)

            # Step 7: Render the stats
            page = pdf.new_page(header=True, footer=True)
            page.append(self.components["STATISTICS"](
                            all_time_returns=format_number(res.all_time_returns),
                            year_returns=format_number(res.year_returns),
                            annual_returns=format_number(res.annual_returns),
                            current_equity=format_number(portfolio_worth.values[-1]),
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
                            yearly_trades=format_number(res.num_trades)))
            page.append(self.components["CASH"](
                                            avail_cash=format_number(state.account.get_cash()),
                                            net_worth=format_number(portfolio_worth.values[-1]),
                                            deposits=format_number(state.cash_controller.monthly_variation(state.timestamp)),
                                            collateral=format_number(state.account.collateral)))
        else:
            page = pdf.new_page(header=True, footer=True)
            page.append(self.components["STATISTICS"](
                           all_time_returns="0",
                           year_returns="0",
                           annual_returns="0",
                           current_equity="0",
                           avg_drawdown="0",
                           avg_trade="0",
                           avg_trade_dur="0",
                           best_trade="0",
                           calmar_ratio="0",
                           equity_peak="0",
                           index_returns="0",
                           max_drawdown="0",
                           max_trade_dur="0",
                           min_trade_dur="0",
                           profit_factor="0",
                           sharp_ratio="0",
                           sortino_ratio="0",
                           sqn="0",
                           win_rate="0",
                           worst_trade="0",
                           yearly_exits="0",
                           yearly_trades="0"))
            page.append(self.components["CASH"](avail_cash=format_number(state.account.get_cash()),
                           net_worth=format_number(state.account.get_cash()),
                           deposits=format_number(state.cash_controller._total_deposited), collateral="0"))

        # Handle css
        css = CSS(string=self.get_style(self.style))

        # Render the whole
        html_text = pdf.render()
        html = HTML(string=html_text)


        # Save the file
        html.write_pdf(
            base_path / 'report.pdf', stylesheets=[css])

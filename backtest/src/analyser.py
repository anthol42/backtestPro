from .metadata import Metadata
from .broker import Broker, StepState
from .account import Account
from .tsData import TSData
from .trade import Trade, TradeOrder, TradeType
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional
import mplfinance as mpf
import matplotlib.pyplot as plt

class Analyser:
    """
    This class is used to analyse the backtest results.
    It has few utility methods that helps the user to visualize the backtest results.
    """
    def __init__(self, data: List[Dict[str, TSData]], backtest_result_state: dict):
        self.broker = Broker.load_state(backtest_result_state["run_states"]["broker"])
        self.account = Account.load_state(backtest_result_state["run_states"]["account"])
        self.data = data
        self.main_timestep = backtest_result_state["metadata"]["backtest_parameters"]["main_timestep"]

        self.metadata = Metadata.load(backtest_result_state["metadata"])
        self.window = self.metadata.backtest_parameters["window"]
        self.strategy_name = backtest_result_state["stats"]["strategy_name"]
        self.initial_cash = backtest_result_state["metadata"]["backtest_parameters"]["initial_cash"]
        self.risk_free_rate = backtest_result_state["metadata"]["backtest_parameters"]["risk_free_rate"]

        self.start = datetime.fromisoformat(backtest_result_state["stats"]["start"])
        self.end = datetime.fromisoformat(backtest_result_state["stats"]["end"])
        self.equity_history = backtest_result_state["run_states"]["equity_history"]
        self.start = datetime.fromisoformat(backtest_result_state["stats"]["start"])
        self.end = datetime.fromisoformat(backtest_result_state["stats"]["end"])
        self.duration = self.end - self.start
        self.exposure_time = timedelta(seconds=float(backtest_result_state["stats"]["exposure_time"]))
        self.equity_final = backtest_result_state["stats"]["equity_final"]
        self.equity_peak = backtest_result_state["stats"]["equity_peak"]
        self.returns = backtest_result_state["stats"]["returns"]
        self.index_returns = backtest_result_state["stats"]["index_returns"]
        self.annual_returns = backtest_result_state["stats"]["annual_returns"]
        self.sharp_ratio = backtest_result_state["stats"]["sharp_ratio"]
        self.sortino_ratio = backtest_result_state["stats"]["sortino_ratio"]
        self.max_drawdown = backtest_result_state["stats"]["max_drawdown"]
        self.avg_drawdown = backtest_result_state["stats"]["avg_drawdown"]
        self.calmar_ratio = backtest_result_state["stats"]["calmar_ratio"]
        self.num_trades = backtest_result_state["stats"]["num_trades"]
        self.win_rate = backtest_result_state["stats"]["win_rate"]
        self.best_trade = backtest_result_state["stats"]["best_trade"]
        self.worst_trade = backtest_result_state["stats"]["worst_trade"]
        self.avg_trade = backtest_result_state["stats"]["avg_trade"]
        self.max_trade_duration = backtest_result_state["stats"]["max_trade_duration"]
        self.avg_trade_duration = backtest_result_state["stats"]["avg_trade_duration"]
        self.min_trade_duration = backtest_result_state["stats"]["min_trade_duration"]
        self.profit_factor = backtest_result_state["stats"]["profit_factor"]
        self.sqn = backtest_result_state["stats"]["sqn"]

    def get_trade_history(self):
        """
        This method returns the trade history of the backtest.
        :return: The trade history
        """
        return self.broker.portfolio.get_trades()

    def get_margin_calls(self) -> List[Dict[str, Union[datetime, dict]]]:
        """
        This method returns the margin calls of the backtest.
        :return: The margin calls
        """
        return [{"timestamp": state.timestamp, "call": state.margin_calls} for state in self.broker.historical_states]

    def get_transactions(self) -> List[Dict[str, Union[datetime, dict]]]:
        """
        This method returns the transactions of the backtest.
        :return: The transactions
        """
        return self.account

    def __getitem__(self, timestamp: datetime):
        """
        This method returns the backtest state prior to the given timestamp including the current timestep.
        :param timestamp: The timestamp to get the account state
        :return: The account state at the given timestamp
        """
        return list(filter(lambda x: x.timestamp <= timestamp, self.broker.historical_states))

    def inspect(self, ticker: str, timestamp: datetime):
        """
        This method returns a TradeInspector object that can be used to visualize the trade order on the chart.
        :param ticker: The ticker to inspect
        :param timestamp: The timestep to inspect
        :return: The TradeInspector object
        """
        bktst_state = self[timestamp][-1]
        # Readjust the timestamp to a real timestep
        timestamp = bktst_state.timestamp
        data = [df[ticker].loc[:timestamp] for df in self.data]
        window_start = data[self.main_timestep].iloc[-self.window].index[0]
        data = [df.iloc[window_start:] for df in data]
        trades = self.broker.portfolio.get_trades()
        trade_order = None
        trade = None
        for t in trades:
            if t.timestamp == timestamp and t.security == ticker:
                trade_order = t.trade_order
                trade = t
                break
        if trade_order is None:
            raise ValueError(f"No trade order found for the given timestamp: {timestamp} and ticker: {ticker}!")
        return TradeInspector(ticker, data, self.main_timestep, trade_order, trade, bktst_state)


    def focus(self, ticker: str):
        """
        This method will return a class that focus on a given ticker.  It will have methods to visualize the trades
        and the indicators for the given ticker.
        :param ticker: The ticker to lookout
        :return: TickerFocus
        """
        data = [df[ticker] for df in self.data]
        trades = [t for t in self.broker.portfolio.get_trades() if t.security == ticker]
        bktst_state = self.broker.historical_states
        return TickerFocus(ticker, data, self.main_timestep, trades, bktst_state, self.window)



class TickerFocus:
    def __init__(self, ticker: str, data: List[TSData], main_timestep: int, trades: List[Trade],
                 bktst_state: List[StepState], window: int):
        self.ticker = ticker
        self.data = data
        self.main_timestep = main_timestep
        self.trades = trades
        self.bktst_state = bktst_state
        self.window = window


    def __str__(self):
        """
        This method returns a string representation of the TickerFocus object.  It will print the trades and the price
        history for the given ticker.  (It also adds the worth and the number of margin calls for each timesteps)
        :return:
        """
        s = f"Ticker: {self.ticker}\n"
        if len(self.bktst_state) == 0:
            return s + "There are no state in the simulation.  It might not have been run."
        if len(self.bktst_state) > 1000:
            for state in self.bktst_state[:500]:
                trade = None
                for t in self.trades:
                    if t.timestamp == state.timestamp:
                        trade = t
                        break
                pricing = self.data[self.main_timestep].loc[state.timestamp]
                open = pricing["Open"]
                high = pricing["High"]
                low = pricing["Low"]
                close = pricing["Close"]
                worth = state.worth
                n_margin_calls = len(state.margin_calls)
                s += self.format_line(state.timestamp, open, high, low, close, worth, n_margin_calls, trade) + "\n"
            s += "...\n"
            for state in self.bktst_state[-500:]:
                trade = None
                for t in self.trades:
                    if t.timestamp == state.timestamp:
                        trade = t
                        break
                pricing = self.data[self.main_timestep].loc[state.timestamp]
                open = pricing["Open"]
                high = pricing["High"]
                low = pricing["Low"]
                close = pricing["Close"]
                worth = state.worth
                n_margin_calls = len(state.margin_calls)
                s += self.format_line(state.timestamp, open, high, low, close, worth, n_margin_calls, trade) + "\n"
        else:
            for state in self.bktst_state:
                trade = None
                for t in self.trades:
                    if t.timestamp == state.timestamp:
                        trade = t
                        break
                pricing = self.data[self.main_timestep].loc[state.timestamp]
                open = pricing["Open"]
                high = pricing["High"]
                low = pricing["Low"]
                close = pricing["Close"]
                worth = state.worth
                n_margin_calls = len(state.margin_calls)
                s += self.format_line(state.timestamp, open, high, low, close, worth, n_margin_calls, trade) + "\n"
        return s


    def format_line(self, timestamp: datetime, open: float, high: float, low: float, close: float, worth: float,
                    n_margin_calls, trade: Trade) -> str:
        trade = f"[X]: {trade.amount + trade.amount_borrowed}" if trade is not None else "[ ]: "
        return (f"{timestamp} -- O: {self.format_float(open, 4)} H: {self.format_float(high, 4)} "
                f"L: {self.format_float(low, 4)} C: {self.format_float(close, 4)} "
                f"W: {self.format_float(worth, 10)} M: {self.format_float(n_margin_calls, 4)} T: {trade}")

    @staticmethod
    def format_float(n: float, length: int) -> str:
        """
        Make sure that the float value is converted as a string and is aligned to the right and with the good len.
        :return: The float as a formattted string
        """
        s = str(n)
        if len(s) > length:
            return s[:length]
        else:
            return " " * (length - len(s)) + s


    def __getitem__(self, timestamp):
        """
        Focus the TickerFocus object to the given timestamp.
        :param timestamp: The timestamp to focus on.
        :return: dict
        """
        data = [ts[self.ticker].loc[:timestamp] for ts in self.data]
        window_start = data[self.main_timestep].iloc[-self.window].index[0]
        data = [ds.loc[window_start:] for ds in data]
        timestamp = data[self.main_timestep].index[0]    # Should have a len of 1
        state = None
        for s in self.bktst_state:
            if s.timestamp == timestamp:
                state = s
                break

        if state is None:
            raise ValueError(f"The given timestamp did not arrived on any steps: {timestamp}!")

        trade = None
        for t in self.trades:
            if t.timestamp == timestamp:
                trade = t
                break

        return {
            "timestamp": timestamp,
            "data": data,
            "state": state,
            "trade": trade
        }





class TradeInspector:
    def __init__(self, ticker: str, data: List[TSData], main_timestep: int, trade_order: TradeOrder, trade: Trade, bktst_state: StepState):
        self.ticker = ticker
        self.data = data
        self.main_timestep = main_timestep
        self.trade_order = trade_order
        self.trade = trade
        self.bktst_state = bktst_state

    def plot(self, indicator_cb: callable = lambda x: []):
        """
        This method plots the trade order on the chart with the price history (Only in the window)
        This helps visualize what the strategy saw to make that decision.
        LEGEND:
            Green: Enter a Position
            Red: Exit a Position
            ^: Buy
            v: Sell
        :param indicator_cb: The indicator callback to plot on the chart.  Returns a list of 'make_addplot' objects
        :return: None
        """
        if self.trade is None:
            signal_arr = np.full(len(self.data), np.nan)
            if self.trade_order.trade_type == TradeType.BuyLong:
                signal_arr[-1] = self.trade.security_price
                signal = mpf.make_addplot(signal_arr, type="scatter", markersize=100, marker="^", color="g")
            elif self.trade.trade_type == TradeType.SellLong:
                signal_arr[-1] = self.trade.security_price
                signal = mpf.make_addplot(signal_arr, type="scatter", markersize=100, marker="v", color="r")
            elif self.trade.trade_type == TradeType.SellShort:
                signal_arr[-1] = self.trade.security_price
                signal = mpf.make_addplot(signal_arr, type="scatter", markersize=100, marker="v", color="g")
            elif self.trade.trade_type == TradeType.BuyShort:
                signal_arr[-1] = self.trade.security_price
                signal = mpf.make_addplot(signal_arr, type="scatter", markersize=100, marker="^", color="r")
            else:
                raise ValueError(f"Invalid Trade Type!  Got {self.trade.trade_type}, but expect: {TradeType.available()}")

            indicators = indicator_cb(self.data)
            mpf.plot(self.data, type="candle", style="yahoo", volume=True, title=self.ticker,
                     addplot=[signal] + indicators + self.get_indicators())
            plt.show()
        else:
            raise ValueError("No trade found for the given timestamp!  There was only an order.")

    def get_indicators(self):
        """
        This method returns the indicators for the trade order.  It is meant to be override by the user.
        :return: indicators as 'make_addplot' objects.
        """
        return []
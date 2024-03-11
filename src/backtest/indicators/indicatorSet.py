import pandas as pd
from .indicator import Indicator
from typing import List, Union, Dict, Optional



class IndicatorSet:
    def __init__(self, *indicators: Indicator, streaming: bool = False):
        """
        :param indicators: Indicators passed as arguments
        :param streaming: Whether to use the streaming capabilities of the indicators or not.  (Note that if true,
                            indicators will have the choice to use it or not.)  Streaming can improve backtest run time.
        """
        self._indicators: List[Indicator] = list(indicators)
        self._streaming = streaming

    def add(self, indicators: Union[Indicator, List[Indicator]]):
        """
        Add an indicator or list of indicators to the IndicatorSet.
        :param indicators: A single indicator or a list of indicators
        :return: None
        """
        if isinstance(indicators, list):
            self._indicators.extend(indicators)
        else:
            self._indicators.append(indicators)
        self._indicators.append(indicators)

    def run_all(self, data: pd.DataFrame, previous_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run all the indicators in the IndicatorSet on the data and return the results as a DataFrame.  It concatenates
        the input data with the results of the indicators.  (The index should be a datetime index)
        :param data: The input data.  At least an OHLCV chart. [Open, High, Low, Close, Volume]
        :param data: A dataframe containing the previously calculated values of the indicators.  (Used when streaming)
                    If streaming is False, this parameter is ignored.  This should be a dataframe with the same index
                    as the data.  This means that the indicators points that needs to be calculated should be nan, and
                    one already calculated should be the previously calculated values.  The columns should be the output
                    columns names of the indicators.
        :return: The data concatenated with the indicator results
        """
        out = [data]
        for indicator in self._indicators:
            out.append(indicator.get(data, previous_data))
        out_df = pd.concat(out, axis=1)
        return out_df

    @property
    def indicators(self) -> Dict[str, Union[Indicator, List[Indicator]]]:
        """
        Get the indicators in the IndicatorSet as a dictionary with the indicator name as the key.
        IF there are multiple indicators with the same name, the value will be a list of indicators.
        This can happen when an indicator is used multiple times with different parameters. (Example: SMA)
        :return: The dictionary of indicators
        """
        out = {}
        for indicator in self._indicators:
            if indicator.name in out:
                if isinstance(out[indicator.name], list):
                    out[indicator.name].append(indicator)
                else:
                    out[indicator.name] = [out[indicator.name], indicator]
            else:
                out[indicator.name] = indicator
        return out


    def __str__(self):
        s = "IndicatorSet(\n"
        for indicator in self._indicators:
            s += f"    {indicator}\n"
        s += ")"
        return s

    def __repr__(self):
        return f"IndicatorSet(n_indicators={len(self._indicators)})"

    def __len__(self):
        """
        :return: Return the number of indicators in the IndicatorSet.
        """
        return len(self._indicators)

    def __iter__(self):
        """
        :return: Return an iterator on the indicators in the IndicatorSet.
        """
        return iter(self._indicators)

    def toList(self) -> List[str]:
        """
        Convert the IndicatorSet to a list of string representations of the indicators.
        :return: List of string representations of the indicators
        """
        return [str(indicator) for indicator in self._indicators]
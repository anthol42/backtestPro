import pandas as pd
from .indicator import Indicator
from typing import List, Union, Dict



class IndicatorSet:
    def __init__(self, *indicators: Indicator):
        self._indicators: List[Indicator] = list(indicators)

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

    def run_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all the indicators in the IndicatorSet on the data and return the results as a DataFrame.  It concatenates
        the input data with the results of the indicators.  (The index should be a datetime index)
        :param data: The input data.  At least an OHLCV chart. [Open, High, Low, Close, Volume]
        :return: The data concatenated with the indicator results
        """
        out = [data]
        for indicator in self._indicators:
            out.append(indicator.get(data))
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


    def toList(self) -> List[str]:
        """
        Convert the IndicatorSet to a list of string representations of the indicators.
        :return: List of string representations of the indicators
        """
        return [str(indicator) for indicator in self._indicators]
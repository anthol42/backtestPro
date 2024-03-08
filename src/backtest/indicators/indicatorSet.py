import pandas as pd
from .indicator import Indicator
from typing import List, Union, Dict



class IndicatorSet:
    def __init__(self, *indicators: Indicator):
        self._indicators: List[Indicator] = list(indicators)

    def add(self, indicators: Union[Indicator, List[Indicator]]):
        if isinstance(indicators, list):
            self._indicators.extend(indicators)
        else:
            self._indicators.append(indicators)
        self._indicators.append(indicators)

    def run_all(self, data: pd.DataFrame) -> pd.DataFrame:
        out = [data]
        for indicator in self._indicators:
            out.append(indicator.get(data))
        out_df = pd.concat(out, axis=1)
        return out_df

    @property
    def indicators(self) -> Dict[str, Indicator]:
        return {indicator.name: indicator for indicator in self._indicators}
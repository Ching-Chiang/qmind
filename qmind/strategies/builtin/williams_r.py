"""
Williams %R Strategy.

%R < -80  → oversold  → enter_long
%R > -20  → overbought → enter_short
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("williams_r", "Williams %R 超买超卖")
class WilliamsRStrategy(BaseStrategy):
    """Williams %R 均值回归策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        lbp = self.params.get("lookback_period", 14)
        df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", -80)
        overbought = self.params.get("overbought", -20)
        # %R 从下方穿越 -80 → 超卖 → 做多
        df["enter_long"] = (
            (df["williams_r"] < oversold) &
            (df["williams_r"].shift(1) >= oversold)
        )
        # %R 从上方穿越 -20 → 超买 → 做空
        df["enter_short"] = (
            (df["williams_r"] > overbought) &
            (df["williams_r"].shift(1) <= overbought)
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["williams_r"] > -50
        df["exit_short"] = df["williams_r"] < -50
        return df

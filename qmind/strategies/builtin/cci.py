"""
Commodity Channel Index (CCI) Strategy.

CCI > +100 → overbought → enter_short
CCI < -100 → oversold  → enter_long
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("cci", "CCI 超买超卖")
class CCIStrategy(BaseStrategy):
    """CCI 均值回归策略 — 极端值反转交易"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cci_period = self.params.get("cci_period", 20)
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], cci_period)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", -100)
        overbought = self.params.get("overbought", 100)
        # CCI 从下方穿越 +100 → 超买 → 做空
        df["enter_short"] = (
            (df["cci"] > overbought) &
            (df["cci"].shift(1) <= overbought)
        )
        # CCI 从上方穿越 -100 → 超卖 → 做多
        df["enter_long"] = (
            (df["cci"] < oversold) &
            (df["cci"].shift(1) >= oversold)
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["cci"] > 0
        df["exit_short"] = df["cci"] < 0
        return df

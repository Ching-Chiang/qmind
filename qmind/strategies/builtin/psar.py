"""
Parabolic SAR (PSAR) 趋势跟踪策略。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("psar", "Parabolic SAR 趋势跟踪")
class PSARStrategy(BaseStrategy):
    """Parabolic SAR 趋势跟踪策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        step = self.params.get("step", 0.02)
        max_step = self.params.get("max_step", 0.2)
        # psar_up is non-NaN during uptrends, psar_down during downtrends
        psar_up = ta.trend.psar_up(df["high"], df["low"], df["close"], step=step, max_step=max_step)
        psar_down = ta.trend.psar_down(df["high"], df["low"], df["close"], step=step, max_step=max_step)
        df["psar"] = psar_up.combine_first(psar_down)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # enter_long: price crosses above SAR (uptrend starts)
        df["enter_long"] = (
            (df["close"] > df["psar"]) &
            (df["close"].shift(1) <= df["psar"].shift(1))
        )
        # enter_short: price crosses below SAR (downtrend starts)
        df["enter_short"] = (
            (df["close"] < df["psar"]) &
            (df["close"].shift(1) >= df["psar"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # exit_long when close drops below SAR
        df["exit_long"] = df["close"] < df["psar"]
        # exit_short when close rises above SAR
        df["exit_short"] = df["close"] > df["psar"]
        return df

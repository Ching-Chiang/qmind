"""
双均线交叉策略 — 经典趋势跟踪。

快速均线上穿慢速均线 → 做多
快速均线下穿慢速均线 → 做空
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("ma_cross", "双均线交叉 — 经典趋势跟踪")
class MACrossStrategy(BaseStrategy):
    """双均线交叉策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.params.get("fast_period", 9)
        slow = self.params.get("slow_period", 21)
        df["sma_fast"] = ta.trend.sma_indicator(df["close"], fast)
        df["sma_slow"] = ta.trend.sma_indicator(df["close"], slow)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["enter_long"] = (df["sma_fast"] > df["sma_slow"]) & (df["sma_fast"].shift(1) <= df["sma_slow"].shift(1))
        df["enter_short"] = (df["sma_fast"] < df["sma_slow"]) & (df["sma_fast"].shift(1) >= df["sma_slow"].shift(1))
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["sma_fast"] < df["sma_slow"]
        df["exit_short"] = df["sma_fast"] > df["sma_slow"]
        return df


@register_strategy("ma_cross_triple", "三均线交叉 — 中期趋势确认")
class TripleMACrossStrategy(BaseStrategy):
    """三均线交叉策略"""
    timeframe = "4h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma_10"] = ta.trend.sma_indicator(df["close"], 10)
        df["sma_30"] = ta.trend.sma_indicator(df["close"], 30)
        df["sma_60"] = ta.trend.sma_indicator(df["close"], 60)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned_up = (df["sma_10"] > df["sma_30"]) & (df["sma_30"] > df["sma_60"])
        cross = (df["sma_10"] > df["sma_30"]) & (df["sma_10"].shift(1) <= df["sma_30"].shift(1))
        df["enter_long"] = aligned_up & cross
        df["enter_short"] = False  # 三均线只做多
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = (df["sma_10"] < df["sma_30"]) | (df["sma_30"] < df["sma_60"])
        df["exit_short"] = False
        return df

"""
三均线趋势跟随策略。

快线(9) / 中线(21) / 慢线(55) 完全排列对齐时入场。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("triple_ema", "三均线趋势跟随")
class TripleEMAStrategy(BaseStrategy):
    """三均线多头/空头排列趋势跟随策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.params.get("fast_period", 9)
        medium = self.params.get("medium_period", 21)
        slow = self.params.get("slow_period", 55)

        df["ema_fast"] = ta.trend.ema_indicator(df["close"], fast)
        df["ema_medium"] = ta.trend.ema_indicator(df["close"], medium)
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], slow)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 多头排列：快 > 中 > 慢
        df["enter_long"] = (
            (df["ema_fast"] > df["ema_medium"]) &
            (df["ema_medium"] > df["ema_slow"]) &
            # 避免重复信号：上一根 K 线尚未完全排列
            ~(
                (df["ema_fast"].shift(1) > df["ema_medium"].shift(1)) &
                (df["ema_medium"].shift(1) > df["ema_slow"].shift(1))
            )
        )
        # 空头排列：快 < 中 < 慢
        df["enter_short"] = (
            (df["ema_fast"] < df["ema_medium"]) &
            (df["ema_medium"] < df["ema_slow"]) &
            ~(
                (df["ema_fast"].shift(1) < df["ema_medium"].shift(1)) &
                (df["ema_medium"].shift(1) < df["ema_slow"].shift(1))
            )
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 排列被破坏时平仓
        df["exit_long"] = df["ema_fast"] < df["ema_medium"]
        df["exit_short"] = df["ema_fast"] > df["ema_medium"]
        return df

"""
量价突破策略。

价格创新高/新低的同时成交量显著放大，确认突破有效性。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("volume_breakout", "量价突破")
class VolumeBreakoutStrategy(BaseStrategy):
    """价格突破支撑/阻力位 + 成交量放量确认"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.get("lookback", 20)
        volume_mult = self.params.get("volume_multiplier", 1.5)

        df["highest_high"] = df["high"].rolling(lookback).max()
        df["lowest_low"] = df["low"].rolling(lookback).min()
        df["avg_volume"] = df["volume"].rolling(lookback).mean()
        df["volume_threshold"] = df["avg_volume"] * volume_mult
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格突破前高 + 成交量放大
        df["enter_long"] = (
            (df["close"] > df["highest_high"]) &
            (df["close"].shift(1) <= df["highest_high"].shift(1)) &
            (df["volume"] > df["volume_threshold"])
        )
        # 价格跌破前低 + 成交量放大
        df["enter_short"] = (
            (df["close"] < df["lowest_low"]) &
            (df["close"].shift(1) >= df["lowest_low"].shift(1)) &
            (df["volume"] > df["volume_threshold"])
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格回落至平均线或成交量萎缩时平仓
        df["exit_long"] = df["close"] < df["close"].rolling(10).mean()
        df["exit_short"] = df["close"] > df["close"].rolling(10).mean()
        return df

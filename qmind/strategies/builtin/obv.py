"""
On-Balance Volume (OBV) 趋势策略。
OBV 短期均线上穿/下穿长期均线生成信号。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("obv", "OBV 量能趋势")
class OBVStrategy(BaseStrategy):
    """On-Balance Volume 趋势跟踪策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["obv_sma_fast"] = df["obv"].rolling(self.params.get("fast_period", 5)).mean()
        df["obv_sma_slow"] = df["obv"].rolling(self.params.get("slow_period", 20)).mean()
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # enter_long: fast SMA crosses above slow SMA (OBV uptrend)
        df["enter_long"] = (
            (df["obv_sma_fast"] > df["obv_sma_slow"]) &
            (df["obv_sma_fast"].shift(1) <= df["obv_sma_slow"].shift(1))
        )
        # enter_short: fast SMA crosses below slow SMA (OBV downtrend)
        df["enter_short"] = (
            (df["obv_sma_fast"] < df["obv_sma_slow"]) &
            (df["obv_sma_fast"].shift(1) >= df["obv_sma_slow"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # exit_long when OBV turns down (fast crosses below slow)
        df["exit_long"] = df["obv_sma_fast"] < df["obv_sma_slow"]
        # exit_short when OBV turns up (fast crosses above slow)
        df["exit_short"] = df["obv_sma_fast"] > df["obv_sma_slow"]
        return df

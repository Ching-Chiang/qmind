"""
ATR-Based Stop Loss Strategy.

Uses ATR for dynamic stop placement and exit signals.
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("atr_stop", "ATR 动态止损")
class ATRStopStrategy(BaseStrategy):
    """ATR 动态止损策略 — 趋势跟踪 + ATR 波动率出场"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        atr_period = self.params.get("atr_period", 14)
        sma_period = self.params.get("sma_period", 20)
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], atr_period
        )
        df["sma_20"] = ta.trend.sma_indicator(df["close"], sma_period)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格在 SMA 上方 → uptrend → 做多
        df["enter_long"] = (
            (df["close"] > df["sma_20"]) &
            (df["close"].shift(1) <= df["sma_20"].shift(1))
        )
        # 价格在 SMA 下方 → downtrend → 做空
        df["enter_short"] = (
            (df["close"] < df["sma_20"]) &
            (df["close"].shift(1) >= df["sma_20"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        atr_mult = self.params.get("atr_mult", 2.0)
        # 价格跌破 入场价(用 SMA 近似) - ATR 倍数 → 止损离场
        df["exit_long"] = df["close"] < (df["sma_20"] - atr_mult * df["atr"])
        # 价格涨破 入场价(用 SMA 近似) + ATR 倍数 → 止损离场
        df["exit_short"] = df["close"] > (df["sma_20"] + atr_mult * df["atr"])
        return df

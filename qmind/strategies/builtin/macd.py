"""
MACD 金叉死叉策略。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("macd", "MACD 金叉死叉")
class MACDStrategy(BaseStrategy):
    """MACD 金叉/死叉策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        df["macd_diff"] = df["macd"] - df["macd_signal"]
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["enter_long"] = (df["macd_diff"] > 0) & (df["macd_diff"].shift(1) <= 0)
        df["enter_short"] = (df["macd_diff"] < 0) & (df["macd_diff"].shift(1) >= 0)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["macd_diff"] < 0
        df["exit_short"] = df["macd_diff"] > 0
        return df


@register_strategy("macd_rsi", "MACD+RSI 双重确认")
class MACDRSIStrategy(BaseStrategy):
    """MACD + RSI 双重确认 — 减少假信号"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        df["macd_diff"] = df["macd"] - df["macd_signal"]
        df["rsi"] = ta.momentum.rsi(df["close"], 14)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        macd_buy = (df["macd_diff"] > 0) & (df["macd_diff"].shift(1) <= 0)
        rsi_oversold = df["rsi"] < 40
        df["enter_long"] = macd_buy & rsi_oversold

        macd_sell = (df["macd_diff"] < 0) & (df["macd_diff"].shift(1) >= 0)
        rsi_overbought = df["rsi"] > 60
        df["enter_short"] = macd_sell & rsi_overbought
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["macd_diff"] < 0
        df["exit_short"] = df["macd_diff"] > 0
        return df

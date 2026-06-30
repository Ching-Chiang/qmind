"""
RSI 超买超卖策略。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("rsi", "RSI 超买超卖")
class RSIStrategy(BaseStrategy):
    """RSI 超买超卖均值回归策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.momentum.rsi(df["close"], 14)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        df["enter_long"] = (df["rsi"] < oversold) & (df["rsi"].shift(1) >= oversold)
        df["enter_short"] = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["rsi"] > 50
        df["exit_short"] = df["rsi"] < 50
        return df


@register_strategy("bollinger", "布林带突破")
class BollingerStrategy(BaseStrategy):
    """布林带突破策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bb_high"] = ta.volatility.bollinger_hband(df["close"])
        df["bb_mid"] = ta.volatility.bollinger_mavg(df["close"])
        df["bb_low"] = ta.volatility.bollinger_lband(df["close"])
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格突破上轨且布林带扩张 -> 做多
        df["enter_long"] = (
            (df["close"] > df["bb_high"]) &
            (df["close"].shift(1) <= df["bb_high"].shift(1)) &
            (df["bb_width"] > df["bb_width"].shift(5))
        )
        # 价格跌破下轨 -> 做空
        df["enter_short"] = (
            (df["close"] < df["bb_low"]) &
            (df["close"].shift(1) >= df["bb_low"].shift(1)) &
            (df["bb_width"] > df["bb_width"].shift(5))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["close"] < df["bb_mid"]
        df["exit_short"] = df["close"] > df["bb_mid"]
        return df


@register_strategy("donchian", "唐奇安通道突破")
class DonchianStrategy(BaseStrategy):
    """唐奇安通道突破策略"""
    timeframe = "1d"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 20)
        df["donchian_high"] = df["high"].rolling(period).max()
        df["donchian_low"] = df["low"].rolling(period).min()
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["enter_long"] = (df["close"] > df["donchian_high"]) & (df["close"].shift(1) <= df["donchian_high"].shift(1))
        df["enter_short"] = (df["close"] < df["donchian_low"]) & (df["close"].shift(1) >= df["donchian_low"].shift(1))
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["close"] < df["donchian_mid"]
        df["exit_short"] = df["close"] > df["donchian_mid"]
        return df

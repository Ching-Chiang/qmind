"""
ADX+MACD 组合策略。

趋势强度（ADX > 25）+ 趋势方向（MACD 金叉/死叉）双重确认。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("adx_macd", "ADX+MACD 趋势组合")
class ADXMACDStrategy(BaseStrategy):
    """ADX 趋势强度 + MACD 方向确认的复合策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_period = self.params.get("adx_period", 14)
        macd_fast = self.params.get("macd_fast", 12)
        macd_slow = self.params.get("macd_slow", 26)
        macd_signal = self.params.get("macd_signal", 9)

        # ADX 趋势强度
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], adx_period)

        # MACD 指标
        macd_indicator = ta.trend.MACD(
            df["close"],
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal,
        )
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_threshold = self.params.get("adx_threshold", 25)

        # ADX > 25 且 MACD 上穿信号线
        df["enter_long"] = (
            (df["adx"] > adx_threshold) &
            (df["macd_diff"] > 0) &
            (df["macd_diff"].shift(1) <= 0)
        )
        # ADX > 25 且 MACD 下穿信号线
        df["enter_short"] = (
            (df["adx"] > adx_threshold) &
            (df["macd_diff"] < 0) &
            (df["macd_diff"].shift(1) >= 0)
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # ADX 跌破 20（趋势消失）或 MACD 反转
        df["exit_long"] = (df["adx"] < 20) | (df["macd_diff"] < 0)
        df["exit_short"] = (df["adx"] < 20) | (df["macd_diff"] > 0)
        return df

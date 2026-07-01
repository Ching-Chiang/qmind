"""
Ichimoku Cloud 策略。
自包含计算：Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span。
"""

from __future__ import annotations

import pandas as pd

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("ichimoku", "Ichimoku Cloud 突破")
class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud 趋势跟踪策略"""
    timeframe = "1d"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        tenkan_period = self.params.get("tenkan_period", 9)
        kijun_period = self.params.get("kijun_period", 26)
        senkou_b_period = self.params.get("senkou_b_period", 52)

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        df["tenkan_sen"] = (
            df["high"].rolling(tenkan_period).max() +
            df["low"].rolling(tenkan_period).min()
        ) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        df["kijun_sen"] = (
            df["high"].rolling(kijun_period).max() +
            df["low"].rolling(kijun_period).min()
        ) / 2

        # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, shifted forward 26
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(kijun_period)

        # Senkou Span B: (52-period high + 52-period low) / 2, shifted forward 26
        df["senkou_span_b"] = (
            (df["high"].rolling(senkou_b_period).max() +
             df["low"].rolling(senkou_b_period).min()) / 2
        ).shift(kijun_period)

        # Chikou Span: close shifted back 26
        df["chikou_span"] = df["close"].shift(-kijun_period)

        # Cloud thickness (used for filtering)
        df["cloud_top"] = df[["senkou_span_a", "senkou_span_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_span_a", "senkou_span_b"]].min(axis=1)
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # enter_long: price above cloud AND Tenkan-sen > Kijun-sen (bullish TK cross)
        df["enter_long"] = (
            (df["close"] > df["cloud_top"]) &
            (df["tenkan_sen"] > df["kijun_sen"]) &
            (df["tenkan_sen"].shift(1) <= df["kijun_sen"].shift(1))
        )
        # enter_short: price below cloud AND Tenkan-sen < Kijun-sen (bearish TK cross)
        df["enter_short"] = (
            (df["close"] < df["cloud_bottom"]) &
            (df["tenkan_sen"] < df["kijun_sen"]) &
            (df["tenkan_sen"].shift(1) >= df["kijun_sen"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # exit_long when close breaks below cloud bottom
        df["exit_long"] = df["close"] < df["cloud_bottom"]
        # exit_short when close breaks above cloud top
        df["exit_short"] = df["close"] > df["cloud_top"]
        return df

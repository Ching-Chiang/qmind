"""
市场数据工具 — K 线/深度/资金费率查询。

工具函数遵循 JSON Schema 定义，供 LLM Agent 调用。
"""

from __future__ import annotations

from typing import Any

from qmind.graph.state import MarketData


async def get_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 200,
    source: str = "auto",
) -> MarketData:
    """获取 K 线数据"""
    from qmind.data.sources.factory import DataSourceFactory
    factory = DataSourceFactory()
    return await factory.fetch_market_data(symbol, source=source, interval=interval)


async def calculate_indicators(market_data: MarketData) -> dict[str, Any]:
    """计算技术指标"""
    klines = market_data.klines.get(list(market_data.klines.keys())[0], [])
    if len(klines) < 50:
        return {"error": "数据不足，至少需要 50 根 K 线"}

    import pandas as pd
    import ta

    closes = pd.Series([k.close for k in klines])
    highs = pd.Series([k.high for k in klines])
    lows = pd.Series([k.low for k in klines])
    volumes = pd.Series([k.volume for k in klines])

    indicators: dict[str, Any] = {
        "sma_20": float(ta.trend.sma_indicator(closes, 20).iloc[-1]) if len(closes) >= 20 else None,
        "sma_50": float(ta.trend.sma_indicator(closes, 50).iloc[-1]) if len(closes) >= 50 else None,
        "sma_200": float(ta.trend.sma_indicator(closes, 200).iloc[-1]) if len(closes) >= 200 else None,
        "ema_12": float(ta.trend.ema_indicator(closes, 12).iloc[-1]) if len(closes) >= 12 else None,
        "ema_26": float(ta.trend.ema_indicator(closes, 26).iloc[-1]) if len(closes) >= 26 else None,
        "rsi_14": float(ta.momentum.rsi(closes, 14).iloc[-1]) if len(closes) >= 14 else None,
        "macd": float(ta.trend.macd(closes).iloc[-1]) if len(closes) >= 26 else None,
        "macd_signal": float(ta.trend.macd_signal(closes).iloc[-1]) if len(closes) >= 26 else None,
        "bb_high": float(ta.volatility.bollinger_hband(closes).iloc[-1]) if len(closes) >= 20 else None,
        "bb_mid": float(ta.volatility.bollinger_mavg(closes).iloc[-1]) if len(closes) >= 20 else None,
        "bb_low": float(ta.volatility.bollinger_lband(closes).iloc[-1]) if len(closes) >= 20 else None,
        "atr_14": (
            float(ta.volatility.average_true_range(highs, lows, closes, 14).iloc[-1])
            if len(closes) >= 14 else None
        ),
        "volume_sma_20": float(volumes.rolling(20).mean().iloc[-1]) if len(volumes) >= 20 else None,
        "obv": float(ta.volume.on_balance_volume(closes, volumes).iloc[-1]) if len(closes) >= 1 else None,
    }
    return indicators

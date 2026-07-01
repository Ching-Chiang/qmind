"""
符号格式转换 — 统一不同数据源/交易所的符号表示。

用户输入: BTC/USDT
YFinance: BTC-USD
Binance: BTC/USDT
Tushare: 000001.SZ
"""

from __future__ import annotations

import re


def to_yfinance(symbol: str) -> str:
    """转换为 yfinance 格式"""
    s = symbol.upper().strip()
    # BTC/USDT → BTC-USD
    s = re.sub(r"/", "-", s)
    # BTC-USD → BTC-USD (去掉 USDT 中的 T)
    s = re.sub(r"-USDT$", "-USD", s)
    return s


def to_binance(symbol: str) -> str:
    """转换为 Binance 格式"""
    s = symbol.upper().strip()
    if "/" not in s:
        # BTCUSDT → BTC/USDT
        for quote in ["USDT", "USD", "BUSD", "BTC", "ETH"]:
            if s.endswith(quote) and s != quote:
                base = s[: -len(quote)]
                s = f"{base}/{quote}"
                break
    return s


def to_tushare(symbol: str) -> str:
    """转换为 Tushare 格式"""
    s = symbol.upper().strip()
    # 000001 → 000001.SZ
    if re.match(r"^\d{6}$", s):
        first = int(s[0])
        if first in (0, 3, 4):
            return f"{s}.SZ"
        elif first == 6:
            return f"{s}.SH"
    return s


def detect_source(symbol: str) -> str:
    """自动检测数据源"""
    s = symbol.upper().strip()
    # 纯数字 → A 股 (Tushare)
    if re.match(r"^\d{6}$", s):
        return "tushare"
    # 带 .SZ/.SH → A 股
    if s.endswith((".SZ", ".SH", ".BJ")):
        return "tushare"
    # crypto → Binance
    if any(c in s for c in ["/", "-"]) and any(q in s for q in ["USDT", "USD", "BTC"]):
        return "binance"
    # 美股
    if s.endswith(("US", ".N", ".O")):
        return "yfinance"
    return "yfinance"  # default

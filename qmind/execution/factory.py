"""
交易所工厂 — 统一创建交易所实例。
"""

from __future__ import annotations

from qmind.execution.base import ExchangeBase
from qmind.execution.cex.binance import BinanceExchange
from qmind.execution.dry_run import DryRunExchange


class ExchangeFactory:
    """交易所工厂"""

    @staticmethod
    def create(
        name: str = "dry_run",
        dry_run: bool = True,
        config: dict | None = None,
    ) -> ExchangeBase:
        config = config or {}

        if name == "dry_run" or dry_run:
            return DryRunExchange(
                initial_balance=config.get("initial_balance", 10000.0),
            )
        elif name == "binance":
            return BinanceExchange(
                api_key=config.get("api_key", ""),
                api_secret=config.get("api_secret", ""),
                dry_run=False,
                testnet=config.get("testnet", True),
            )
        elif name == "okx":
            from qmind.execution.cex.okx import OKXExchange
            return OKXExchange(
                api_key=config.get("api_key", ""),
                api_secret=config.get("api_secret", ""),
                passphrase=config.get("passphrase", ""),
                dry_run=False,
                testnet=config.get("testnet", True),
            )
        elif name == "bybit":
            from qmind.execution.cex.bybit import BybitExchange
            return BybitExchange(
                api_key=config.get("api_key", ""),
                api_secret=config.get("api_secret", ""),
                dry_run=False,
            )
        else:
            raise ValueError(f"Unknown exchange: {name}")

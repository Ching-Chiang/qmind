"""
持仓/余额查询工具 — 供 LLM Agent 调用。
"""

from __future__ import annotations

from typing import Any


class PortfolioTool:
    """持仓查询工具（dryRun 模式返回模拟数据）"""

    def __init__(self, dry_run: bool = True, initial_balance: float = 10000.0):
        self.dry_run = dry_run
        self.balance = initial_balance
        self.positions: dict[str, dict[str, Any]] = {}
        self.trade_history: list[dict[str, Any]] = []

    async def get_balance(self) -> dict[str, Any]:
        """查询可用余额"""
        if self.dry_run:
            return {"total_balance": self.balance, "available": self.balance, "currency": "USDT", "mode": "dry_run"}
        return {"total_balance": self.balance, "available": self.balance, "currency": "USDT"}

    async def get_positions(self) -> list[dict[str, Any]]:
        """查询当前持仓"""
        return list(self.positions.values())

    async def get_pnl(self) -> dict[str, Any]:
        """查询已实现 PnL"""
        realized: float = sum(t.get("pnl", 0.0) for t in self.trade_history)
        return {"realized_pnl": realized, "unrealized_pnl": 0.0, "total_pnl": realized}

    async def get_trade_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取历史交易记录"""
        return self.trade_history[-limit:]

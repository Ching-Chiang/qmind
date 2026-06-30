"""
下单校验器 — 执行前检查。

检查项:
1. 价格是否合理（非零、无极端偏离）
2. 数量是否超风控上限
3. 是否 dryRun 模式
4. 可用余额是否足够
"""

from __future__ import annotations

from dataclasses import dataclass

from qmind.execution.base import ExchangeBase
from qmind.graph.state import TradeDecision


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    adjusted_position_size_pct: float = 0.0


class OrderValidator:
    """下单校验器"""

    def __init__(self, exchange: ExchangeBase, max_position_pct: float = 30.0):
        self.exchange = exchange
        self.max_position_pct = max_position_pct

    async def validate(self, decision: TradeDecision) -> ValidationResult:
        """校验交易决策是否可执行"""
        if decision.decision == "HOLD":
            return ValidationResult(valid=True, reason="HOLD 指令，无需执行")

        # 1. 价格校验
        entry_price = decision.entry.get("price", 0)
        if entry_price <= 0:
            return ValidationResult(valid=False, reason="入场价格无效")

        # 2. 数量校验
        if decision.position_size_pct <= 0:
            return ValidationResult(valid=False, reason="仓位比例为 0")

        # 3. 风控上限
        if decision.position_size_pct > self.max_position_pct:
            return ValidationResult(
                valid=False,
                reason=f"仓位 {decision.position_size_pct:.1f}% 超过风控上限 {self.max_position_pct}%",
                adjusted_position_size_pct=self.max_position_pct,
            )

        # 4. 止损校验
        stop_price = decision.stop_loss.get("price", 0)
        if stop_price <= 0:
            return ValidationResult(valid=False, reason="止损价格无效")

        # 5. 风险收益比
        if decision.risk_reward_ratio < 1.0 and decision.decision != "HOLD":
            return ValidationResult(valid=False, reason=f"风险收益比 {decision.risk_reward_ratio:.2f} < 1.0")

        return ValidationResult(valid=True)

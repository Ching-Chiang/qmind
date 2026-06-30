"""
交易结果评估器 — 评估已完成的交易。

计算:
- PnL (盈亏金额 + 百分比)
- 持仓时长
- 最大浮亏 (MAE) / 最大浮盈 (MFE)
- 滑点 vs 预期
- 执行质量
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qmind.graph.state import Lesson, TradeEvaluation


@dataclass
class TradeRecord:
    """一笔已经结的交易记录"""
    trade_id: str
    symbol: str
    decision: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    position_size: float
    entry_time: datetime
    exit_time: datetime
    stop_loss: float | None = None
    take_profit: list[float] = None
    highest_price: float | None = None  # MFE
    lowest_price: float | None = None   # MAE
    slippage_bps: float = 0.0
    is_dry_run: bool = True

    def __post_init__(self):
        if self.take_profit is None:
            self.take_profit = []


class TradeEvaluator:
    """交易结果评估器"""

    def evaluate(self, trade: TradeRecord) -> TradeEvaluation:
        """评估一笔交易的结果"""
        # PnL 计算
        if trade.decision == "LONG":
            pnl_abs = (trade.exit_price - trade.entry_price) * trade.position_size
            pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
        else:  # SHORT
            pnl_abs = (trade.entry_price - trade.exit_price) * trade.position_size
            pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100

        # 持仓时长
        duration = trade.exit_time - trade.entry_time
        hours = duration.total_seconds() / 3600
        if hours < 1:
            hold_duration = f"{duration.total_seconds() / 60:.0f}m"
        elif hours < 24:
            hold_duration = f"{hours:.1f}h"
        else:
            hold_duration = f"{hours / 24:.1f}d"

        # MAE / MFE
        if trade.highest_price and trade.lowest_price:
            if trade.decision == "LONG":
                mae = (trade.lowest_price - trade.entry_price) / trade.entry_price * 100
                mfe = (trade.highest_price - trade.entry_price) / trade.entry_price * 100
            else:
                mae = (trade.entry_price - trade.highest_price) / trade.entry_price * 100
                mfe = (trade.entry_price - trade.lowest_price) / trade.entry_price * 100
        else:
            mae = 0.0
            mfe = 0.0

        # 执行质量
        exec_quality = self._assess_execution(trade, pnl_abs)

        # 教训占位 — CVRF 反思后填充
        lessons: list[Lesson] = []

        return TradeEvaluation(
            pnl_abs=round(pnl_abs, 2),
            pnl_pct=round(pnl_pct, 4),
            hold_duration=hold_duration,
            mae=round(mae, 4),
            mfe=round(mfe, 4),
            slippage=trade.slippage_bps,
            execution_quality=exec_quality,
            lessons=lessons,
        )

    def _assess_execution(self, trade: TradeRecord, pnl: float) -> str:
        """评估执行质量"""
        if trade.slippage_bps > 10:
            return "差 — 滑点过大"
        if trade.stop_loss and trade.stop_loss >= trade.exit_price:
            if trade.decision == "LONG" and trade.exit_price <= trade.stop_loss:
                return "按止损执行"
            if trade.decision == "SHORT" and trade.exit_price >= trade.stop_loss:
                return "按止损执行"
        if pnl > 0:
            return "良好"
        return "完成"

"""
CVRF 完整闭环流水线。

回测 → 逐笔总结教训 → 存入记忆库 → 下一窗口自动读取
"""

from __future__ import annotations

import logging
from datetime import datetime

from qmind.graph.state import MarketConditionVector, MemoryEntry
from qmind.learning.cvrf import CVRFReflection
from qmind.learning.evaluator import TradeEvaluator, TradeRecord
from qmind.learning.memory import MemoryStore

logger = logging.getLogger(__name__)


class CVRFPipeline:
    """CVRF 学习闭环流水线"""

    def __init__(self, reflection: CVRFReflection, memory: MemoryStore):
        self.reflection = reflection
        self.memory = memory
        self.evaluator = TradeEvaluator()

    async def process_trade(
        self,
        trade: TradeRecord,
        analysis_summary: str = "",
        market_context: str = "",
    ) -> MemoryEntry:
        """处理一笔交易：评估 → 反思 → 存入记忆库"""
        # 1. 评估交易结果
        evaluation = self.evaluator.evaluate(trade)
        logger.info(
            f"交易 {trade.trade_id}: {evaluation.pnl_pct:+.2f}% "
            f"| MAE: {evaluation.mae:.2f}% | MFE: {evaluation.mfe:.2f}%"
        )

        # 2. CVRF 反思
        lessons = await self.reflection.reflect(
            trade, evaluation, analysis_summary, market_context,
        )
        evaluation.lessons = lessons

        # 3. 提取市况特征
        market_condition = await self.reflection.extract_market_condition(trade)

        # 4. 构建记忆条目
        entry = MemoryEntry(
            symbol=trade.symbol,
            timestamp=datetime.utcnow(),
            market_condition=market_condition,
            lessons=lessons,
            trade_outcome={
                "trade_id": trade.trade_id,
                "pnl_pct": evaluation.pnl_pct,
                "pnl_abs": evaluation.pnl_abs,
                "mae": evaluation.mae,
                "mfe": evaluation.mfe,
                "decision": trade.decision,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
            },
            was_bull_correct=(evaluation.pnl_pct > 0) if trade.decision == "LONG" else (evaluation.pnl_pct < 0),
            was_bear_correct=(evaluation.pnl_pct > 0) if trade.decision == "SHORT" else (evaluation.pnl_pct < 0),
            embedding=self._compute_embedding(market_condition),
        )

        # 5. 存入记忆库
        entry_id = self.memory.save(entry)
        entry.id = entry_id
        logger.info(f"教训已存入记忆库 (id={entry_id}, {len(lessons)} 条)")

        return entry

    def _compute_embedding(self, condition: MarketConditionVector) -> list[float]:
        """计算市况特征向量的 embedding"""
        trend_map = {"uptrend": 1.0, "downtrend": -1.0, "sideways": 0.0, "reversal": 0.5, "": 0.0}
        vol_map = {"low": 0.0, "medium": 0.5, "high": 1.0, "": 0.5}
        cycle_map = {"accumulation": 0.0, "markup": 0.33, "distribution": 0.66, "markdown": 1.0, "": 0.5}
        vol_trend_map = {"increasing": 1.0, "decreasing": -1.0, "flat": 0.0, "": 0.0}

        return [
            trend_map.get(condition.trend, 0.0),
            vol_map.get(condition.volatility, 0.5),
            cycle_map.get(condition.market_cycle, 0.5),
            max(-1.0, min(1.0, condition.momentum)),
            vol_trend_map.get(condition.volume_trend, 0.0),
        ]

    async def batch_process(
        self,
        trades: list[TradeRecord],
        analyses: dict[str, str] = None,
    ) -> list[MemoryEntry]:
        """批量处理多笔交易"""
        entries = []
        for trade in trades:
            try:
                entry = await self.process_trade(
                    trade,
                    analysis_summary=(analyses or {}).get(trade.trade_id, ""),
                )
                entries.append(entry)
            except Exception as e:
                logger.error(f"处理交易 {trade.trade_id} 失败: {e}")
        return entries

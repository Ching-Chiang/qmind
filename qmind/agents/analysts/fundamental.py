"""
基本面分析师 — 财务数据、估值、行业对比。

注意：加密货币/外汇等品种的基本面分析可能受限。
"""

from __future__ import annotations

from qmind.agents.analysts.base import BaseAnalyst
from qmind.agents.protocol import FundamentalReport
from qmind.graph.state import AnalystReport, MarketData

FUNDAMENTAL_SYSTEM_PROMPT = """你是一位基本面分析师，关注资产的长期价值和内在驱动因素。

你的任务是分析可获取的基本面数据，输出结构化基本面报告。

请关注:
1. 估值水平 (P/E、P/B、DCF 等)
2. 盈利能力 (ROE、利润率、营收增长)
3. 行业地位与竞争优势
4. 增长前景与催化剂
5. 潜在风险 (监管、竞争、宏观经济)

如果数据不足（如加密货币），请明确说明并调整置信度。
输出严格符合 JSON Schema。"""


class FundamentalAnalyst(BaseAnalyst):
    """基本面分析师"""

    @property
    def analyst_name(self) -> str:
        return "fundamental"

    @property
    def system_prompt(self) -> str:
        return FUNDAMENTAL_SYSTEM_PROMPT

    async def analyze(self, market_data: MarketData) -> AnalystReport:
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        prompt = (
            f"# 基本面分析\n\n"
            f"标的: {market_data.symbol}\n"
            f"当前价格: {current_price:.2f}\n"
            f"数据时间: {market_data.as_of or 'N/A'}\n\n"
            f"请分析该标的的基本面情况。由于数据限制，请基于公开可获取的信息进行分析。\n"
            f"如果信息不足，请明确说明并降低置信度。\n"
        )

        result = await self.parser.parse(
            prompt, FundamentalReport,
            system=self.system_prompt, temperature=self.temperature,
        )
        return AnalystReport(
            analyst=result.analyst,
            stance=result.stance,
            confidence=result.confidence,
            core_reason=result.core_reason,
            key_signals=[s.model_dump() for s in result.key_signals],
            risk_factors=result.risk_factors,
            details=result.financial_health,
        )

"""
情绪分析师 — 市场情绪、资金流向、持仓变化。

参考 FinGPT 情感分析思路。
"""

from __future__ import annotations

from qmind.agents.analysts.base import BaseAnalyst
from qmind.agents.protocol import SentimentReport
from qmind.graph.state import AnalystReport, MarketData

SENTIMENT_SYSTEM_PROMPT = """你是一位市场情绪分析师，专注于市场参与者的情绪和资金流向。

你的任务是分析市场情绪数据，输出结构化的情绪分析报告。

请关注:
1. 多空比率 (Long/Short Ratio)
2. 资金费率 (Funding Rate) — 永续合约市场情绪指标
3. 持仓量变化 (Open Interest 增减)
4. 社交媒体/新闻情绪
5. 大户/鲸鱼动向
6. 期权市场数据 (Put/Call Ratio) — 如有

⚠️ 注意: 极端情绪往往是反向指标。所有人都看多时反而是风险信号。
输出严格符合 JSON Schema。"""


class SentimentAnalyst(BaseAnalyst):
    """市场情绪分析师"""

    @property
    def analyst_name(self) -> str:
        return "sentiment"

    @property
    def system_prompt(self) -> str:
        return SENTIMENT_SYSTEM_PROMPT

    async def analyze(self, market_data: MarketData) -> AnalystReport:
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        prompt = (
            f"# 市场情绪分析\n\n"
            f"标的: {market_data.symbol}\n"
            f"当前价格: {current_price:.2f}\n"
            f"资金费率: {market_data.funding_rate if market_data.funding_rate is not None else 'N/A'}\n"
            f"持仓量: {market_data.open_interest if market_data.open_interest is not None else 'N/A'}\n"
            f"数据时间: {market_data.as_of or 'N/A'}\n\n"
            f"请基于上述数据和你的市场知识分析当前市场情绪状态。\n"
            f"如果数据有限，请基于当前价格行为和交易量推导市场情绪，并标注不确定性。\n"
        )

        result = await self.parser.parse(
            prompt, SentimentReport,
            system=self.system_prompt, temperature=self.temperature,
        )
        details_parts = [
            f"多空比: {result.long_short_ratio or 'N/A'}",
            f"资金费率: {result.funding_rate or 'N/A'}",
            f"持仓量变化: {result.open_interest_change or 'N/A'}",
        ]
        return AnalystReport(
            analyst=result.analyst,
            stance=result.stance,
            confidence=result.confidence,
            core_reason=result.core_reason,
            key_signals=[s.model_dump() for s in result.key_signals],
            risk_factors=result.risk_factors,
            details=" | ".join(details_parts),
        )

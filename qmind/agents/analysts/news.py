"""
宏观/新闻分析师 — 政策、宏观经济、重大事件。
"""

from __future__ import annotations

from qmind.agents.analysts.base import BaseAnalyst
from qmind.agents.protocol import NewsReport
from qmind.graph.state import AnalystReport, MarketData

NEWS_SYSTEM_PROMPT = """你是一位宏观分析师，专注于宏观经济政策和重大事件对市场的影响。

你的任务是分析宏观环境，输出结构化的宏观分析报告。

请关注:
1. 货币政策 (利率、QT/QE、央行表态)
2. 经济数据 (CPI、GDP、PMI、就业)
3. 地缘政治风险
4. 监管政策变化
5. 行业重大新闻

如果无具体新闻数据输入，请基于你对当前宏观经济环境的了解进行分析，
并明确标注哪些是事实、哪些是基于事实的推断。
输出严格符合 JSON Schema。"""


class NewsAnalyst(BaseAnalyst):
    """宏观/新闻分析师"""

    @property
    def analyst_name(self) -> str:
        return "news"

    @property
    def system_prompt(self) -> str:
        return NEWS_SYSTEM_PROMPT

    async def analyze(self, market_data: MarketData) -> AnalystReport:
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        news_summary = ""
        if market_data.news:
            for i, n in enumerate(market_data.news[:5]):
                news_summary += f"{i+1}. {n.get('title', n.get('content', 'N/A'))}\n"

        prompt = (
            f"# 宏观/新闻分析\n\n"
            f"标的: {market_data.symbol}\n"
            f"当前价格: {current_price:.2f}\n"
            f"数据时间: {market_data.as_of or 'N/A'}\n\n"
            f"## 新闻数据\n"
            f"{news_summary or '无具体新闻输入，请基于宏观环境知识进行分析。'}\n\n"
            f"请分析这些因素对该标的的潜在影响。\n"
        )

        result = await self.parser.parse(prompt, NewsReport, system=self.system_prompt, temperature=self.temperature)
        return AnalystReport(
            analyst=result.analyst,
            stance=result.stance,
            confidence=result.confidence,
            core_reason=result.core_reason,
            key_signals=[s.model_dump() for s in result.key_signals],
            risk_factors=result.risk_factors,
            details=result.policy_impact,
        )

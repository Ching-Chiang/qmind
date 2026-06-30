"""
分析师并行调度器 — 4 个分析师同时执行，单分析师超时不影响整体。
"""

from __future__ import annotations

import asyncio
from typing import Any

from qmind.agents.analysts.fundamental import FundamentalAnalyst
from qmind.agents.analysts.news import NewsAnalyst
from qmind.agents.analysts.sentiment import SentimentAnalyst
from qmind.agents.analysts.technical import TechnicalAnalyst
from qmind.graph.state import AnalystReport, MarketData
from qmind.llm.client import LLMClient


class AnalystRunner:
    """分析师并行调度器"""

    def __init__(self, llm_client: LLMClient, timeout: float = 60.0):
        self.timeout = timeout
        # 异构 LLM 配置 — 防回声室
        self.analysts = [
            TechnicalAnalyst(llm_client, model="claude-sonnet-4-6", temperature=0.3),
            FundamentalAnalyst(llm_client, model="gpt-4o", temperature=0.4),
            SentimentAnalyst(llm_client, model="deepseek-chat", temperature=0.3),
            NewsAnalyst(llm_client, model="claude-sonnet-4-6", temperature=0.5),
        ]

    async def run_all(self, market_data: MarketData) -> list[AnalystReport]:
        """并行运行所有分析师"""
        async def safe_analyze(analyst: Any) -> AnalystReport | None:
            try:
                return await asyncio.wait_for(
                    analyst.analyze(market_data),
                    timeout=self.timeout,
                )
            except TimeoutError:
                return AnalystReport(
                    analyst=analyst.analyst_name,
                    stance="neutral",
                    confidence=0.0,
                    core_reason=f"分析师超时 (>{self.timeout}s)",
                )
            except Exception as e:
                return AnalystReport(
                    analyst=analyst.analyst_name,
                    stance="neutral",
                    confidence=0.0,
                    core_reason=f"分析师异常: {e}",
                )

        tasks = [safe_analyze(a) for a in self.analysts]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

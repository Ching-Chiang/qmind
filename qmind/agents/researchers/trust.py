"""
Trust Agent — 证据锚定 + 风险识别。

验证决策是否可以追溯到具体数据，防止 LLM 产生幻觉。
辩论中不做方向判断，只输出风险点。
"""

from __future__ import annotations

from qmind.agents.protocol import DebateResultProtocol
from qmind.graph.state import AnalystReport, MarketData
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser

TRUST_SYSTEM_PROMPT = """你是一个 Trust Agent（信任审计员）。

你的任务:
1. 检查每个分析报告中的论断是否有具体数据支持
2. 识别缺乏数据支持的"幻觉论断"
3. 评估分析师的置信度是否与其论据充分程度匹配
4. 标记你认为风险最高的 1-2 个问题

⚠️ 重要: 你不需要判断方向（看多/看空），只需要验证论据质量。
输出严格符合 JSON Schema。"""


class TrustAgent:
    """信任审计 Agent — 验证论据质量"""

    def __init__(self, llm_client: LLMClient, model: str = "deepseek-chat"):
        self.parser = StructuredParser(client=llm_client, model=model, caller="trust_agent")

    async def verify(
        self,
        market_data: MarketData,
        analyses: list[AnalystReport],
    ) -> dict[str, list[str]]:
        """验证分析报告的论据质量"""
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        prompt = (
            f"## Trust Agent 验证\n\n"
            f"标的: {market_data.symbol} | 当前价格: {current_price:.2f}\n\n"
            f"### 分析师报告\n"
        )
        for a in analyses:
            prompt += (
                f"\n[{a.analyst.upper()}] 立场: {a.stance} | "
                f"置信度: {a.confidence:.2f}\n"
                f"核心逻辑: {a.core_reason}\n"
                f"风险因素: {', '.join(a.risk_factors) if a.risk_factors else '无'}\n"
            )

        prompt += (
            "\n请逐一评价每个分析师:\n"
            "1. 论据是否充分支撑其立场？\n"
            "2. 置信度是否合理？\n"
            "3. 哪些论断缺乏数据支持？\n"
            "4. 最重要的 1-2 个风险点是什么？\n"
        )

        result = await self.parser.parse(prompt, DebateResultProtocol, system=TRUST_SYSTEM_PROMPT)
        return {
            "assessment": result.final_assessment,
            "concerns": result.disagreement_points,
        }

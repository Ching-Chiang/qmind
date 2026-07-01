"""
Skeptic Agent — 漏洞识别，方向不改变。

Prompt 明确: 不改变方向，只输出风险点。
基于论文审阅结论: 辩论不做方向判断，只做风险降级。
"""

from __future__ import annotations

from qmind.agents.protocol import DebateResultProtocol
from qmind.graph.state import AnalystReport, MarketData
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser

SKEPTIC_SYSTEM_PROMPT = """你是一个 Skeptic Agent（怀疑论者）。

⚠️ 你的目标: 找出决策中的漏洞和盲区，而不是改变方向。

具体任务:
1. 审查各方论据中可能存在的逻辑漏洞
2. 识别被忽略的风险因素
3. 找出数据解读中的可能偏差
4. 评估"如果市场走势相反"会怎样

约束:
- ❌ 不要建议改变交易方向
- ❌ 不要输出你自己的交易建议
- ✅ 只输出: 风险点、漏洞、被忽视的因素

记住: 好的风控不是避免交易，而是知道哪里可能出错。"""


class SkepticAgent:
    """Skeptic Agent — 只找漏洞，不换方向"""

    def __init__(self, llm_client: LLMClient, model: str = "deepseek-chat"):
        self.parser = StructuredParser(client=llm_client, model=model, caller="skeptic_agent")

    async def scrutinize(
        self,
        market_data: MarketData,
        analyses: list[AnalystReport],
        debate_assessment: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """审查决策中的漏洞"""
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        prompt = (
            f"## Skeptic 审查\n\n"
            f"标的: {market_data.symbol} | 当前价格: {current_price:.2f}\n\n"
            f"### 分析师共识\n"
        )
        for a in analyses:
            prompt += f"- [{a.analyst}] {a.stance} ({a.confidence:.0%}): {a.core_reason}\n"

        concerns = debate_assessment.get('concerns', [])
        prompt += (
            f"\n### Trust Agent 已标注的问题\n"
            f"{debate_assessment.get('assessment', '无')}\n"
        )
        prompt += f"{'风险点: ' + ', '.join(concerns) if concerns else '无'}\n\n"
        prompt += (
            "请从怀疑论者角度找出:\n"
            "1. 还有哪些盲区？\n"
            "2. 分析师忽略了什么？\n"
            "3. 最坏情景是什么？\n\n"
            "⚠️ 再次强调: 不需要建议方向，只需指出风险。\n"
        )

        result = await self.parser.parse(prompt, DebateResultProtocol, system=SKEPTIC_SYSTEM_PROMPT)
        return {
            "gaps": result.disagreement_points,
            "worst_case": result.final_assessment,
        }

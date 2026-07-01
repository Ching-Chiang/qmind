"""
Debate Leader — 辩论主导者。

输出置信度降级因子和仓位缩减比例，不做方向判断。
基于论文审阅: 辩论 = 偏差校正，不是 alpha 生成。
"""

from __future__ import annotations

from qmind.agents.protocol import DebateResultProtocol
from qmind.graph.state import AnalystReport, MarketData
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser

LEADER_SYSTEM_PROMPT = """你是一个 Debate Leader（辩论主导者）。

你的职责是整合所有分析意见和风险审查，输出:
1. 置信度降级因子 (0.0-1.0): 风险越大降级越多，1.0=不降，0.0=完全取消
2. 仓位缩减比例 (0.0-1.0): 建议缩减多少仓位，0.0=不缩减，1.0=全减
3. 最终评估: 总结关键风险点和共识点

关键原则:
- 不做方向判断（不用管做多还是做空）
- 如果分歧高 → 置信度降级 + 仓位缩减
- 如果风险审查发现严重问题 → 大幅降级
- 保持客观，基于证据而非直觉

输出严格符合 JSON Schema。"""


class DebateLeader:
    """Debate Leader — 输出置信度降级和仓位缩减"""

    def __init__(self, llm_client: LLMClient, model: str = "deepseek-chat"):
        self.parser = StructuredParser(client=llm_client, model=model, caller="debate_leader")

    async def lead(
        self,
        market_data: MarketData,
        analyses: list[AnalystReport],
        disagreement: dict,
        trust_assessment: dict[str, list[str]],
        skeptic_assessment: dict[str, list[str]],
    ) -> DebateResultProtocol:
        """主导辩论，输出降级因子"""
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        prompt = (
            f"## Debate Leader 综合评估\n\n"
            f"标的: {market_data.symbol} | 当前价格: {current_price:.2f}\n\n"
            f"### 分析师分歧\n"
            f"分歧度 δ: {disagreement.get('delta', 0)}\n"
            f"级别: {disagreement.get('level', 'unknown')}\n"
            f"最强分析师: {disagreement.get('strongest_analyst', 'N/A')} "
            f"({disagreement.get('strongest_stance', 'N/A')}, "
            f"置信度: {disagreement.get('strongest_confidence', 0):.2f})\n\n"
            f"### 各分析师立场\n"
        )
        for a in analyses:
            prompt += f"- [{a.analyst}] {a.stance} ({a.confidence:.0%}): {a.core_reason}\n"

        prompt += (
            f"\n### Trust Agent 审计\n{trust_assessment.get('assessment', 'N/A')}\n"
            f"风险点: {', '.join(trust_assessment.get('concerns', []))}\n\n"
            f"### Skeptic 审查\n"
            f"漏洞: {', '.join(skeptic_assessment.get('gaps', []))}\n"
            f"最坏情景: {skeptic_assessment.get('worst_case', 'N/A')}\n\n"
            f"请基于上述信息输出:\n"
            f"1. 置信度降级因子 — 考虑到所有风险后，置信度应打几折？\n"
            f"2. 仓位缩减比例 — 仓位应缩减多少？\n"
            f"3. 最终评估 — 关键风险和共识总结\n"
        )

        return await self.parser.parse(prompt, DebateResultProtocol, system=LEADER_SYSTEM_PROMPT)

"""
CVRF 反思引擎 — LLM 用自然语言总结"学到了什么"。

基于 FINCON 论文的 Concept Verbal Reinforcement Learning:
传统 RL 需要数千次迭代，CVRF 仅需 4 个 episode 即有明显提升。

核心 prompt 模板:
"这是一笔 {盈利/亏损} 的交易。
当初做决策时我们认为：{原分析}
辩论中多方说：{多方论点}  空方说：{空方论点}
实际发生了什么：{市场走势}
请你回答：
1. 当时哪个判断是对的？哪个是错的？
2. 空方提出的风险哪些应验了？
3. 从这笔交易中学到的 3 条教训是什么？
4. 下次遇到类似市况应该注意什么？"
"""

from __future__ import annotations

from pydantic import BaseModel

from qmind.graph.state import Lesson, MarketConditionVector, TradeEvaluation
from qmind.learning.evaluator import TradeRecord
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser


class CVRFLessons(BaseModel):
    lessons: list[Lesson]

CVRF_SYSTEM_PROMPT = """你是一个专业的交易复盘专家 (CVRF — Conceptual Verbal Reinforcement Learning)。

你的任务是从每一笔交易中提取可操作的教训，让系统在未来类似市况中做出更好的决策。

请从以下维度分析:
1. 判断准确性 — 哪个判断对了？哪个错了？为什么？
2. 风险管理 — 止损位置是否合理？仓位是否合适？
3. 辩论评估 — 多空双方各自提出了哪些论点？哪些应验了？
4. 执行质量 — 入场/出场时机是否最优？
5. 核心教训 — 最有价值的 1-3 条可迁移教训

输出结构化教训 JSON，每条教训包含:
- lesson: 具体的自然语言教训
- confidence: 你对该教训确信程度的 0-1 评分
- source: 来源分类 (entry/exit/risk/sizing/market_timing)"""

CVRF_REFLECTION_PROMPT = """## CVRF 交易反思

### 交易信息
- 标的: {symbol}
- 方向: {decision}
- 入场价: {entry_price}
- 出场价: {exit_price}
- 持仓时长: {hold_duration}
- 盈亏: {pnl_pct:.2f}% ({pnl_abs:.2f})
- 最大浮亏: {mae:.2f}%
- 最大浮盈: {mfe:.2f}%
- 滑点: {slippage} bps

### 当初决策依据
{analysis_summary}

### 实际市场走势
{market_context}

---

请回答以下 4 个问题:
1. 当时哪个判断是对的？哪个是错的？
2. 辩论/分析中提出的风险哪些应验了？
3. 从这笔交易中学到的 1-3 条核心教训是什么？
4. 下次遇到类似市况应该注意什么？

请输出结构化的 Lessons。"""


class CVRFReflection:
    """CVRF 反思引擎"""

    def __init__(self, llm_client: LLMClient, model: str = "deepseek-chat"):
        self.parser = StructuredParser(
            client=llm_client, model=model, caller="cvrf_reflection",
        )

    async def reflect(
        self,
        trade: TradeRecord,
        evaluation: TradeEvaluation,
        analysis_summary: str = "",
        market_context: str = "",
    ) -> list[Lesson]:
        """对一笔已完成交易进行反思，返回教训列表"""
        prompt = CVRF_REFLECTION_PROMPT.format(
            symbol=trade.symbol,
            decision=trade.decision,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            hold_duration=evaluation.hold_duration,
            pnl_pct=evaluation.pnl_pct,
            pnl_abs=evaluation.pnl_abs,
            mae=evaluation.mae,
            mfe=evaluation.mfe,
            slippage=evaluation.slippage,
            analysis_summary=analysis_summary or "（无详细记录）",
            market_context=market_context or "（无市场背景数据）",
        )

        result = await self.parser.parse(
            prompt, CVRFLessons,
            system=CVRF_SYSTEM_PROMPT,
            temperature=0.5,
        )

        return result.lessons

    async def extract_market_condition(
        self,
        trade: TradeRecord,
    ) -> MarketConditionVector:
        """从交易记录中提取市况特征向量"""
        prompt = (
            f"基于以下交易记录，描述当时的市场状况:\n"
            f"标的: {trade.symbol} | 方向: {trade.decision}\n"
            f"入场: {trade.entry_price} | 出场: {trade.exit_price}\n"
            f"最高: {trade.highest_price} | 最低: {trade.lowest_price}\n\n"
            f"请输出市场状况特征:\n"
            f"- trend (uptrend/downtrend/sideways/reversal)\n"
            f"- volatility (low/medium/high)\n"
            f"- market_cycle (accumulation/markup/distribution/markdown)\n"
            f"- momentum (-1.0 to 1.0)\n"
            f"- volume_trend (increasing/decreasing/flat)\n"
        )

        result = await self.parser.parse(
            prompt, MarketConditionVector,
            temperature=0.3,
        )
        return result



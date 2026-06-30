"""
三角风控审核 — 激进/保守/中立三个角度独立审核。

任一否决 → 取消交易。
全体通过 → 加上 CVaR 硬约束校验 → 进入执行。

基于 FINCON 的 CVaR 公式:
    CVaR(95%) = 历史最差 5% 交易日平均亏损
    如果 当前仓位 x 预期最大波动 > CVaR 阈值 → 强制缩小仓位
"""

from __future__ import annotations

from typing import Any

from qmind.agents.protocol import CVaRCheck, RiskFinalVerdict, RiskReview
from qmind.graph.state import MarketData, TradeDecision
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser

AGGRESSIVE_SYSTEM_PROMPT = """你是一个激进风控审核员。你的倾向是偏向通过。

你的关注点:
1. 这笔交易的机会有多大？错过会后悔吗？
2. 风险是否可控？
3. 仓位是否合理？

你可以批准、修改（建议更激进的仓位）或拒绝（只在极少数情况下）。
记住: 你的角色是机会导向，但仍有否决权。"""

CONSERVATIVE_SYSTEM_PROMPT = """你是一个保守风控审核员。你的倾向是偏向否决。

你的关注点:
1. 最坏情况是什么？能不能承受？
2. 有足够的 margin of safety 吗？
3. 是否应该等待更好的入场点？

你有一票否决权。当不确定性高时，选择拒绝。
记住: 你的角色是保护本金。宁愿错过，不要亏损。"""

NEUTRAL_SYSTEM_PROMPT = """你是一个中立风控审核员。你客观评估风险和收益。

你的关注点:
1. 风险收益比是否合理？(至少 1:2)
2. 有没有被忽略的盲区？
3. 仓位大小是否与置信度匹配？

提供客观的调整建议，不做倾向性判断。"""

CVAR_PROMPT = """基于 FINCON 论文的 CVaR 约束计算。

给定交易决策和市场数据，计算:
1. 当前风险敞口 = 仓位(%) x 账户余额 x ATR(止损距离)
2. CVaR(95%) 阈值 = 历史最差 5% 的平均亏损上限
3. 是否通过: 当前风险敞口 <= CVaR 阈值
"""


class TriangleRiskControl:
    """三角风控审核器"""

    def __init__(self, llm_client: LLMClient):
        self.aggressive_parser = StructuredParser(
            client=llm_client, model="claude-sonnet-4-6", caller="risk_aggressive")
        self.conservative_parser = StructuredParser(
            client=llm_client, model="claude-opus-4-8", caller="risk_conservative")
        self.neutral_parser = StructuredParser(
            client=llm_client, model="claude-sonnet-4-6", caller="risk_neutral")

    def _build_prompt(self, decision: TradeDecision, market_data: MarketData) -> str:
        current_price = 0
        for klines in market_data.klines.values():
            if klines:
                current_price = klines[-1].close
                break

        return (
            f"## 风控审核\n\n"
            f"标的: {decision.symbol}\n"
            f"当前价格: {current_price:.2f}\n"
            f"决策: {decision.decision}\n"
            f"入场: {decision.entry.get('price', 'N/A')}\n"
            f"止损: {decision.stop_loss.get('price', 'N/A')}\n"
            f"止盈目标: {[t.get('price') for t in decision.take_profit]}\n"
            f"仓位: {decision.position_size_pct:.1f}%\n"
            f"置信度: {decision.confidence:.2f}\n"
            f"风险收益比: {decision.risk_reward_ratio:.2f}\n"
            f"时间框架: {decision.time_horizon}\n"
            f"最大可接受亏损: {decision.max_acceptable_loss_pct:.2f}%\n\n"
            f"推理链:\n"
            f"Data: {decision.reasoning_chain.get('data_cot', 'N/A')}\n"
            f"Concept: {decision.reasoning_chain.get('concept_cot', 'N/A')}\n"
            f"Thesis: {decision.reasoning_chain.get('thesis_cot', 'N/A')}\n\n"
            f"请审核该交易决策，输出审核意见。\n"
        )

    async def _review(self, parser: StructuredParser, prompt: str, system_prompt: str) -> RiskReview:
        return await parser.parse(prompt, RiskReview, system=system_prompt, temperature=0.3)

    async def review(
        self,
        decision: TradeDecision,
        market_data: MarketData,
        account_balance: float = 10000.0,
    ) -> RiskFinalVerdict:
        """执行三角风控审核"""
        prompt = self._build_prompt(decision, market_data)

        # 并行执行三个风控审核
        import asyncio
        aggressive, conservative, neutral = await asyncio.gather(
            self._review(self.aggressive_parser, prompt, AGGRESSIVE_SYSTEM_PROMPT),
            self._review(self.conservative_parser, prompt, CONSERVATIVE_SYSTEM_PROMPT),
            self._review(self.neutral_parser, prompt, NEUTRAL_SYSTEM_PROMPT),
        )

        # CVaR 硬约束校验
        cvar = self._calculate_cvar(decision, account_balance)

        approvals = [aggressive, conservative, neutral]
        vetoed_by = []
        adjustments: dict[str, Any] = {}

        approved_count = 0
        for rev in approvals:
            if rev.decision == "reject":
                vetoed_by.append(rev.role)
            else:
                approved_count += 1
                if rev.suggested_position_size_pct is not None:
                    adjustments["position_size_pct"] = min(
                        adjustments.get("position_size_pct", 100),
                        rev.suggested_position_size_pct,
                    )

        # CVaR 约束: 如果没通过，也加入否决
        if not cvar.passed:
            vetoed_by.append("cvar_constraint")

        total_vetoes = len(vetoed_by)

        return RiskFinalVerdict(
            approved=total_vetoes == 0 and cvar.passed,
            veto_count=total_vetoes,
            vetoed_by=vetoed_by,
            adjustments=adjustments,
            aggressive_review=aggressive,
            conservative_review=conservative,
            neutral_review=neutral,
            cvar_check=cvar,
            final_position_size_pct=adjustments.get("position_size_pct", decision.position_size_pct),
            final_stop_loss=None,
        )

    def _calculate_cvar(self, decision: TradeDecision, account_balance: float) -> CVaRCheck:
        """CVaR(95%) 硬约束计算"""
        entry_price = decision.entry.get("price", 0) or 0
        stop_price = decision.stop_loss.get("price", 0) or 0
        if entry_price == 0 or stop_price == 0:
            return CVaRCheck(passed=True, current_exposure=0, cvar_threshold=0, margin=0)

        # 止损距离百分比
        stop_pct = abs(entry_price - stop_price) / entry_price if entry_price else 0
        # 当前风险敞口 = 仓位占比 x 账户余额 x 止损距离
        position_value = account_balance * (decision.position_size_pct / 100)
        current_exposure = position_value * stop_pct

        # CVaR 阈值: 假设账户的 5% 为最大可接受单笔亏损
        cvar_threshold = account_balance * 0.05
        # 如果置信度低，阈值更严格
        if decision.confidence < 0.5:
            cvar_threshold *= 0.5
        elif decision.confidence < 0.7:
            cvar_threshold *= 0.8

        passed = current_exposure <= cvar_threshold
        margin = cvar_threshold - current_exposure

        return CVaRCheck(
            passed=passed,
            current_exposure=round(current_exposure, 2),
            cvar_threshold=round(cvar_threshold, 2),
            margin=round(margin, 2),
            calculation_details=(
                f"仓位:{decision.position_size_pct:.1f}% x 余额:{account_balance:.0f} "
                f"x 止损距离:{stop_pct:.2%} = 敞口:{current_exposure:.2f} "
                f"<= 阈值:{cvar_threshold:.2f} ? {'通过' if passed else '否决'}"
            ),
        )

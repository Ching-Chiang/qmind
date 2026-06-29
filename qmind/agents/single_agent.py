"""
Phase 1 — 单 Agent 强基线。

单一 LLM Agent 综合分析 + 决策，作为 Phase 2 多 Agent 的对比基线。
不使用多角色辩论或风控，证明"在没有多 Agent 的情况下能跑出什么水平"。
"""

from __future__ import annotations

from qmind.agents.single_agent_prompt import SINGLE_AGENT_SYSTEM_PROMPT
from qmind.graph.state import MarketData, TradeDecision
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser
from qmind.tools.market_data import calculate_indicators


class SingleTradingAgent:
    """单 Agent 交易决策引擎"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "claude-sonnet-4-6",
    ):
        self.parser = StructuredParser(
            client=llm_client,
            model=model,
            caller="single_agent",
        )
        self.llm_client = llm_client

    def _format_price_trend(self, market_data: MarketData) -> str:
        """提取价格趋势描述"""
        timeframe = list(market_data.klines.keys())[0] if market_data.klines else "1h"
        klines = market_data.klines.get(timeframe, [])
        if len(klines) < 5:
            return "数据不足"

        recent = klines[-20:] if len(klines) >= 20 else klines
        changes = []
        for i in range(1, min(len(recent), 6)):
            pct = (recent[-i].close - recent[-i - 1].close) / recent[-i - 1].close * 100
            direction = "↑" if pct > 0 else "↓"
            changes.append(f"{direction}{abs(pct):.2f}%")

        high = max(k.close for k in klines[-20:]) if klines else 0
        low = min(k.close for k in klines[-20:]) if klines else 0

        return (
            f"当前: {klines[-1].close:.2f} | "
            f"20期最高: {high:.2f} | 20期最低: {low:.2f} | "
            f"最近5根变化: {' '.join(changes[-5:])}"
        )

    def _get_support_resistance(self, klines: list) -> tuple[str, str]:
        """估算支撑和阻力位"""
        if len(klines) < 20:
            return "N/A", "N/A"

        highs = [k.high for k in klines[-50:]] if len(klines) >= 50 else [k.high for k in klines]
        lows = [k.low for k in klines[-50:]] if len(klines) >= 50 else [k.low for k in klines]

        resistance = sum(highs) / len(highs) + (max(highs) - min(lows)) * 0.236
        support = sum(lows) / len(lows) - (max(highs) - min(lows)) * 0.236

        return f"{support:.2f}", f"{resistance:.2f}"

    def _build_analysis_prompt(self, market_data: MarketData) -> str:
        """构建完整的分析 prompt（含三级推理链）"""
        timeframe = list(market_data.klines.keys())[0] if market_data.klines else "1h"
        klines = market_data.klines.get(timeframe, [])

        current_price = klines[-1].close if klines else 0
        price_trend = self._format_price_trend(market_data)
        support, resistance = self._get_support_resistance(klines)

        # 计算技术指标
        indicators = calculate_indicators(market_data) if klines else {}

        def fmt(key: str, unit: str = "") -> str:
            v = indicators.get(key)
            if v is None:
                return f"{key}: N/A"
            return f"{key}: {v:.2f}{unit}"

        ind_lines = [
            fmt("rsi_14"),
            f"MACD: {indicators.get('macd', 0):.2f} / Signal: {indicators.get('macd_signal', 0):.2f}",
            f"布林带上轨: {indicators.get('bb_high', 0):.2f} / "
            f"中轨: {indicators.get('bb_mid', 0):.2f} / "
            f"下轨: {indicators.get('bb_low', 0):.2f}",
            fmt("atr_14"),
        ]
        ma_lines = {
            "sma_20": fmt("sma_20"),
            "sma_50": fmt("sma_50"),
            "sma_200": fmt("sma_200"),
        }

        volume_info = ""
        if "volume_sma_20" in indicators and indicators["volume_sma_20"]:
            current_vol = klines[-1].volume if len(klines) > 0 else 0
            vol_ratio = current_vol / indicators["volume_sma_20"] if indicators["volume_sma_20"] else 1
            volume_info = (
                f"当前量: {current_vol:.0f} / "
                f"均量(20): {indicators['volume_sma_20']:.0f} / "
                f"量比: {vol_ratio:.2f}"
            )

        prompt = (
            f"# 交易分析任务\n\n"
            f"## 标的信息\n"
            f"标的: {market_data.symbol}\n"
            f"时间框架: {timeframe}\n"
            f"数据时间: {market_data.as_of or 'N/A'}\n\n"
        )

        # Data-CoT section
        prompt += "## Data-CoT: 数据层推理\n\n"
        prompt += "### 价格与趋势\n"
        prompt += f"- 当前价格: {current_price:.2f}\n"
        prompt += f"- 近期趋势: {price_trend}\n"
        prompt += f"- 关键支撑: {support}\n"
        prompt += f"- 关键阻力: {resistance}\n\n"
        prompt += "### 技术指标\n"
        prompt += "\n".join(ind_lines) + "\n\n"
        prompt += "### 均线系统\n"
        prompt += f"  {ma_lines['sma_20']}\n  {ma_lines['sma_50']}\n  {ma_lines['sma_200']}\n\n"
        prompt += f"### 成交量\n{volume_info}\n\n"

        # Concept-CoT section
        prompt += (
            "## Concept-CoT: 概念层推理\n\n"
            "请从以下角度分析当前市场状态:\n"
            "1. 趋势状态（上升/下降/震荡/反转）\n"
            "2. 波动率水平（高/中/低）\n"
            "3. 多空力量对比\n"
            "4. 成交量验证\n"
            "5. 风险收益比\n\n"
        )

        # Thesis-CoT section
        prompt += (
            "## Thesis-CoT: 论点层推理\n\n"
            "基于以上分析，请形成你的交易决策。\n"
            "交易方向需在 LONG / SHORT / HOLD 中选择。\n"
            "如果选择 HOLD，只需给出理由，其他字段可以为空或默认值。\n"
        )

        return prompt

    async def analyze(self, market_data: MarketData) -> TradeDecision:
        """执行一次完整分析，返回结构化决策"""
        prompt = self._build_analysis_prompt(market_data)

        decision = await self.parser.parse(
            prompt=prompt,
            schema=TradeDecision,
            system=SINGLE_AGENT_SYSTEM_PROMPT,
            temperature=0.3,
        )

        # 确保 symbol 字段填充
        if not decision.symbol:
            decision.symbol = market_data.symbol

        return decision

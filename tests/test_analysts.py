"""analysts + runner 单元测试 (mock LLM)"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from qmind.agents.analysts.runner import AnalystRunner
from qmind.agents.analysts.technical import TechnicalAnalyst
from qmind.agents.protocol import TechnicalReport
from qmind.graph.state import OHLCV, AnalystReport, MarketData
from qmind.llm.client import LLMClient


class TestTechnicalAnalyst:
    @pytest.fixture
    def analyst(self):
        return TechnicalAnalyst(LLMClient())

    async def test_empty_klines_returns_report(self, analyst):
        md = MarketData(symbol="TEST/USDT")
        with patch.object(analyst.parser, "parse", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = TechnicalReport(
                analyst="technical", stance="neutral", confidence=0.5,
                core_reason="test", key_signals=[], risk_factors=[],
            )
            result = await analyst.analyze(md)
        assert result.analyst == "technical"
        assert result.stance in ("bullish", "neutral", "bearish")

    async def test_llm_parse_error_returns_fallback(self, analyst):
        """当 StructuredParser 抛出异常时 propagate"""
        md = MarketData(symbol="TEST/USDT", klines={"1h": [
            OHLCV(timestamp=i, open=100, high=101, low=99, close=100, volume=1000)
            for i in range(60)
        ]})
        with patch.object(analyst.parser, "parse", side_effect=Exception("LLM error")), \
                pytest.raises(Exception, match="LLM error"):
            await analyst.analyze(md)


class TestAnalystRunner:
    @pytest.fixture
    def runner(self):
        return AnalystRunner(LLMClient(), timeout=5.0)

    async def test_all_analysts_timeout_return_neutral(self, runner):
        """超时降级：返回中性报告"""
        md = MarketData(symbol="TEST/USDT")
        for a in runner.analysts:
            a.analyze = AsyncMock(side_effect=TimeoutError("timeout"))

        results = await runner.run_all(md)
        assert len(results) == 4
        for r in results:
            assert r.stance == "neutral"
            assert r.confidence == 0.0

    async def test_one_fails_others_succeed(self, runner):
        """部分分析师失败不影响其他"""
        with patch.object(runner.analysts[0], "analyze",
                          side_effect=Exception("oops")):
            for a in runner.analysts[1:]:
                a.analyze = AsyncMock(return_value=AnalystReport(
                    analyst=a.analyst_name, stance="bullish", confidence=0.7, core_reason="test",
                ))
            results = await runner.run_all(MarketData(symbol="T"))
        assert len(results) == 4
        # 第一个返回 neutral 降级
        assert results[0].confidence == 0.0
        # 其他应该正常
        assert any(r.confidence > 0 for r in results)

"""LLM 客户端 单元测试（mock 网络调用）"""

from __future__ import annotations

import pytest

from qmind.llm.client import CostTracker, LLMClient, TokenUsage
from qmind.llm.router import LLMRouter


class TestCostTracker:
    """CostTracker 功能测试"""

    def test_initial_state(self):
        ct = CostTracker()
        assert ct.total_cost() == 0.0
        assert ct.total_tokens() == 0
        assert ct.summary()["call_count"] == 0

    def test_record_cost(self):
        ct = CostTracker()
        ct.record(TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost_usd=0.0075,
            model="claude-sonnet-4-6",
            provider="anthropic",
        ))
        assert ct.total_cost() == 0.0075
        assert ct.total_tokens() == 1500
        assert ct.summary()["call_count"] == 1

    def test_summary_by_model(self):
        ct = CostTracker()
        for _ in range(3):
            ct.record(TokenUsage(
                prompt_tokens=500, completion_tokens=250, total_tokens=750,
                cost_usd=0.003, model="gpt-4o-mini", provider="openai",
            ))
        summary = ct.summary()
        assert summary["call_count"] == 3
        assert summary["by_model"]["gpt-4o-mini"]["calls"] == 3


class TestLLMClient:
    """LLMClient 基本功能测试"""

    def test_get_provider_anthropic(self):
        client = LLMClient()
        assert client.get_provider("claude-sonnet-4-6").value == "anthropic"
        assert client.get_provider("claude-opus-4-8").value == "anthropic"
        assert client.get_provider("claude-haiku-4-5").value == "anthropic"

    def test_get_provider_openai(self):
        client = LLMClient()
        assert client.get_provider("gpt-4o").value == "openai"
        assert client.get_provider("gpt-4o-mini").value == "openai"

    def test_get_provider_deepseek(self):
        client = LLMClient()
        assert client.get_provider("deepseek-chat").value == "deepseek"
        assert client.get_provider("deepseek-reasoner").value == "deepseek"

    def test_get_provider_unknown(self):
        client = LLMClient()
        with pytest.raises(ValueError, match="Unknown model"):
            client.get_provider("unknown-model")

    def test_calculate_cost(self):
        client = LLMClient()
        # claude-sonnet-4-6: $3/M input, $15/M output
        cost = client.calculate_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == round(expected, 8)

    def test_calculate_cost_unknown_model(self):
        client = LLMClient()
        cost = client.calculate_cost("unknown-model", 1000, 500)
        assert cost == 0.0


class TestLLMRouter:
    """LLM Router 基本功能测试"""

    def test_router_config(self):
        client = LLMClient()
        router = LLMRouter(client)
        assert router._config["analysis"]["model"] == "claude-sonnet-4-6"
        assert router._config["tool_call"]["model"] == "gpt-4o-mini"
        assert router._config["risk_conservative"]["model"] == "claude-opus-4-8"

    def test_set_config(self):
        client = LLMClient()
        router = LLMRouter(client)
        router.set_config("analysis", model="gpt-4o", temperature=0.1)
        assert router._config["analysis"]["model"] == "gpt-4o"
        assert router._config["analysis"]["temperature"] == 0.1

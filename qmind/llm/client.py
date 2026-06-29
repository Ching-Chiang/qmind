"""
统一 LLM 调用客户端。

支持 Anthropic / OpenAI / DeepSeek 三供应商，
自动 Token 计数与成本追踪。

用法:
    client = LLMClient()
    response = await client.chat(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.content, response.usage)
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# ──────────────────────────────────────────────
# Cost Tracking
# ──────────────────────────────────────────────

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


@dataclass
class TokenUsage:
    """单次调用的 Token 用量与成本"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""


@dataclass
class LLMResponse:
    """统一响应结构"""
    content: str
    usage: TokenUsage
    raw_response: Any = None
    latency_ms: float = 0.0


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


PROVIDER_MODELS: dict[LLMProvider, list[str]] = {
    LLMProvider.ANTHROPIC: ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    LLMProvider.OPENAI: ["gpt-4o", "gpt-4o-mini"],
    LLMProvider.DEEPSEEK: ["deepseek-chat", "deepseek-reasoner"],
}


# ──────────────────────────────────────────────
# Cost Tracker
# ──────────────────────────────────────────────

@dataclass
class CostRecord:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    caller: str = ""


class CostTracker:
    """LLM 调用成本追踪器"""

    def __init__(self):
        self.records: list[CostRecord] = []

    def record(self, usage: TokenUsage, caller: str = "") -> None:
        self.records.append(CostRecord(
            model=usage.model,
            provider=usage.provider,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            caller=caller,
        ))

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"total_cost_usd": 0.0, "total_tokens": 0, "call_count": 0}

        by_model: dict[str, dict] = {}
        for r in self.records:
            m = r.model
            if m not in by_model:
                by_model[m] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            by_model[m]["calls"] += 1
            by_model[m]["prompt_tokens"] += r.prompt_tokens
            by_model[m]["completion_tokens"] += r.completion_tokens
            by_model[m]["cost_usd"] += r.cost_usd

        return {
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens": self.total_tokens(),
            "call_count": len(self.records),
            "by_model": by_model,
        }


# ──────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────

class LLMClient:
    """统一 LLM 客户端"""

    def __init__(self, cost_tracker: CostTracker | None = None):
        self._anthropic_client: AsyncAnthropic | None = None
        self._openai_client: AsyncOpenAI | None = None
        self._deepseek_client: AsyncOpenAI | None = None
        self.cost_tracker = cost_tracker or CostTracker()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic_client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            self._anthropic_client = AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    @property
    def deepseek(self) -> AsyncOpenAI:
        if self._deepseek_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            self._deepseek_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        return self._deepseek_client

    def get_provider(self, model: str) -> LLMProvider:
        for provider, models in PROVIDER_MODELS.items():
            if model in models:
                return provider
        if model.startswith("claude"):
            return LLMProvider.ANTHROPIC
        if model.startswith(("gpt", "o")):
            return LLMProvider.OPENAI
        if model.startswith("deepseek"):
            return LLMProvider.DEEPSEEK
        raise ValueError(f"Unknown model: {model}")

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        caller: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """统一 chat 调用"""
        start = time.perf_counter()
        provider = self.get_provider(model)

        if provider == LLMProvider.ANTHROPIC:
            result = await self._chat_anthropic(model, messages, system, max_tokens, temperature, **kwargs)
        elif provider == LLMProvider.OPENAI:
            result = await self._chat_openai(model, messages, system, max_tokens, temperature, **kwargs)
        elif provider == LLMProvider.DEEPSEEK:
            result = await self._chat_openai(model, messages, system, max_tokens, temperature,
                                             client=self.deepseek, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        latency = (time.perf_counter() - start) * 1000
        usage = TokenUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
            cost_usd=self.calculate_cost(model, result.get("prompt_tokens", 0), result.get("completion_tokens", 0)),
            model=model,
            provider=provider.value,
        )
        self.cost_tracker.record(usage, caller=caller)

        return LLMResponse(
            content=result.get("content", ""),
            usage=usage,
            raw_response=result,
            latency_ms=latency,
        )

    async def _chat_anthropic(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            api_kwargs["system"] = system
        api_kwargs.update(kwargs)

        response = await self.anthropic.messages.create(**api_kwargs)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return {
            "content": content,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    async def _chat_openai(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        client: AsyncOpenAI | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        client = client or self.openai
        api_messages = list(messages)
        if system:
            api_messages.insert(0, {"role": "system", "content": system})

        response = await client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return {
            "content": response.choices[0].message.content or "",
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        _caller: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式 chat 调用"""
        provider = self.get_provider(model)

        if provider == LLMProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(model, messages, system, max_tokens, temperature, **kwargs):
                yield chunk
        else:
            async for chunk in self._stream_openai(model, messages, system, max_tokens, temperature,
                                                    provider=provider, **kwargs):
                yield chunk

    async def _stream_anthropic(self, model: str, messages: list[dict], system: str | None,
                                 max_tokens: int, temperature: float, **kwargs: Any) -> AsyncIterator[str]:
        api_kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        }
        if system:
            api_kwargs["system"] = system
        api_kwargs.update(kwargs)

        async with self.anthropic.messages.stream(**api_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_openai(self, model: str, messages: list[dict], system: str | None,
                              max_tokens: int, temperature: float, provider: LLMProvider = LLMProvider.OPENAI,
                              **kwargs: Any) -> AsyncIterator[str]:
        client = self.deepseek if provider == LLMProvider.DEEPSEEK else self.openai
        api_messages = list(messages)
        if system:
            api_messages.insert(0, {"role": "system", "content": system})

        stream = await client.chat.completions.create(
            model=model, messages=api_messages, max_tokens=max_tokens,
            temperature=temperature, stream=True, **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

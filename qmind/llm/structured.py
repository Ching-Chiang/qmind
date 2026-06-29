"""
结构化输出封装 — Pydantic 校验 + 自动重试。

用法:
    parser = StructuredParser(model="claude-sonnet-4-6", client=llm_client)
    result = await parser.parse(
        prompt="Analyze this stock",
        schema=AnalystReport,
        max_retries=3,
    )
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from qmind.llm.client import LLMClient, LLMResponse

T = TypeVar("T", bound=BaseModel)


class StructuredParser:
    """Pydantic 结构化输出解析器（JSON mode + 自动重试）"""

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        caller: str = "",
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.caller = caller

    async def parse(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.3,
        retry_count: int = 0,
    ) -> T:
        """发送 prompt 并返回 Pydantic 结构化结果"""
        schema_json = schema.model_json_schema()
        content = (
            f"{prompt}\n\n"
            f"你必须输出符合以下 JSON Schema 的纯 JSON"
            f"（不含 markdown 代码块标记）：\n{schema_json}\n\n"
            f"只输出 JSON，不要其他任何文字。"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        response: LLMResponse = await self.client.chat(
            model=self.model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=4096,
            caller=self.caller,
        )

        raw = response.content.strip()
        # 去除可能的 markdown 代码块标记
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return schema.model_validate(data)
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            if retry_count < self.max_retries:
                return await self.parse(
                    prompt=(
                        f"上次解析失败: {e}\n\n"
                        f"请严格输出合法的 JSON 对象，不要包含任何额外文字。\n\n"
                        f"原始请求: {prompt}"
                    ),
                    schema=schema,
                    system=system,
                    temperature=temperature + 0.1,
                    retry_count=retry_count + 1,
                )
            raise

    async def parse_many(
        self,
        prompts: list[str],
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.3,
    ) -> list[T | None]:
        """并行解析多个 prompt"""
        import asyncio

        tasks = [
            self.parse(p, schema, system=system, temperature=temperature)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parsed: list[T | None] = []
        for r in results:
            if isinstance(r, BaseModel):
                parsed.append(r)
            else:
                parsed.append(None)
        return parsed

"""
双 LLM 路由：推理用强模型，工具调用用快模型。

根据任务类型自动分配最优模型:
- 分析、辩论、决策 -> 强模型 (claude-sonnet-4-6)
- 工具调用、数据提取、简单分类 -> 快模型 (gpt-4o-mini / deepseek-chat)
- 深度推理 -> 最强模型 (claude-opus-4-8，仅复杂风控/辩论收敛判断)
"""

from __future__ import annotations

from typing import Any

from qmind.llm.client import LLMClient, LLMResponse


class LLMRouter:
    """LLM 路由分发器"""

    def __init__(self, client: LLMClient):
        self.client = client
        self._config: dict[str, dict[str, Any]] = {
            "analysis": {"model": "deepseek-chat", "temperature": 0.3, "max_tokens": 4096},
            "debate": {"model": "deepseek-chat", "temperature": 0.5, "max_tokens": 4096},
            "decision": {"model": "deepseek-chat", "temperature": 0.2, "max_tokens": 4096},
            "risk_review": {"model": "deepseek-chat", "temperature": 0.3, "max_tokens": 2048},
            "risk_conservative": {"model": "deepseek-chat", "temperature": 0.2, "max_tokens": 2048},
            "cvar_check": {"model": "deepseek-chat", "temperature": 0.1, "max_tokens": 1024},
            "tool_call": {"model": "deepseek-chat", "temperature": 0.1, "max_tokens": 1024},
            "classification": {"model": "deepseek-chat", "temperature": 0.1, "max_tokens": 512},
            "reflection": {"model": "deepseek-chat", "temperature": 0.3, "max_tokens": 2048},
            "data_extraction": {"model": "deepseek-chat", "temperature": 0.1, "max_tokens": 2048},
            "cvrf_learning": {"model": "deepseek-chat", "temperature": 0.5, "max_tokens": 2048},
        }

    async def route(
        self,
        task_type: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        caller: str = "",
        **overrides: Any,
    ) -> LLMResponse:
        """根据任务类型自动路由到合适的模型"""
        config = dict(self._config.get(task_type, self._config["analysis"]))
        model = overrides.pop("model", config.pop("model"))
        temperature = overrides.pop("temperature", config.pop("temperature"))
        max_tokens = overrides.pop("max_tokens", config.pop("max_tokens"))

        return await self.client.chat(
            model=model,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            caller=caller or task_type,
            **config,
            **overrides,
        )

    def set_config(self, task_type: str, **kwargs: Any) -> None:
        """自定义某任务类型的模型/参数"""
        if task_type not in self._config:
            self._config[task_type] = {}
        self._config[task_type].update(kwargs)

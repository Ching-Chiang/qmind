"""
分析师基类 — 所有分析师统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qmind.graph.state import AnalystReport, MarketData
from qmind.llm.client import LLMClient
from qmind.llm.structured import StructuredParser


class BaseAnalyst(ABC):
    """分析师基类"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.3,
    ):
        self.parser = StructuredParser(
            client=llm_client,
            model=model,
            caller=self.analyst_name,
        )
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

    @property
    @abstractmethod
    def analyst_name(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    async def analyze(self, market_data: MarketData) -> AnalystReport:
        ...

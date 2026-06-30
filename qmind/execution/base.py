"""
执行层基类 — 交易所统一接口。

所有交易所实现以下接口，由 ExchangeFactory 统一创建。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class OrderResult:
    """下单结果"""
    order_id: str
    symbol: str
    side: str  # buy / sell
    type: str  # market / limit
    price: float
    quantity: float
    status: str  # open / filled / cancelled / rejected
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    raw: dict[str, Any] = None


@dataclass
class Balance:
    """账户余额"""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # long / short
    quantity: float
    entry_price: float
    mark_price: float
    pnl_unrealized: float
    leverage: int = 1


class ExchangeBase(ABC):
    """交易所基类"""

    def __init__(self, name: str, dry_run: bool = True):
        self.name = name
        self.dry_run = dry_run

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        ...

    @abstractmethod
    async def get_balance(self, asset: str = "") -> list[Balance]:
        """获取余额"""
        ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        **kwargs: Any,
    ) -> OrderResult:
        """下单"""
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """撤单"""
        ...

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> OrderResult:
        """查单"""
        ...

    @abstractmethod
    async def get_positions(self, symbol: str = "") -> list[Position]:
        """获取持仓"""
        ...

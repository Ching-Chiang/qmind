"""
策略注册表 — @register_strategy 装饰器 + 按名称查找。
"""

from __future__ import annotations

from typing import Any

from qmind.strategies.base import BaseStrategy

_registry: dict[str, type[BaseStrategy]] = {}
_loaded = False


def register_strategy(name: str, description: str = ""):
    """策略注册装饰器"""
    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        cls.name = name
        cls.description = description or cls.__doc__ or ""
        _registry[name] = cls
        return cls
    return decorator


def _ensure_loaded() -> None:
    """惰性加载内置策略，避免循环导入"""
    global _loaded
    if not _loaded:
        import qmind.strategies.builtin  # noqa: F401
        _loaded = True


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """按名称获取策略实例"""
    _ensure_loaded()
    cls = _registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_registry.keys())}")
    instance = cls()
    instance.params = params
    return instance


def list_strategies() -> list[dict[str, str]]:
    """列出所有已注册策略"""
    _ensure_loaded()
    return [
        {"name": name, "description": cls.description}
        for name, cls in _registry.items()
    ]

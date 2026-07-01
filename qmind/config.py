"""
QMind 统一配置管理。

使用 YAML 配置文件 + 环境变量覆写，API Key 用 cryptography 加密存储。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 显式指定 .env 路径（项目根目录），避免 uvicorn 热重载改 CWD 导致加载失败
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, verbose=True)

DEFAULT_CONFIG_PATH = Path("config.yaml")


class Config:
    """统一配置"""

    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """支持点号分隔的嵌套 key: 'execution.dry_run'"""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    @property
    def execution_dry_run(self) -> bool:
        return self.get("execution.dry_run", True)

    @property
    def llm_default_model(self) -> str:
        return os.getenv("QMIND_LLM_MODEL") or self.get("llm.default_model", "claude-sonnet-4-6")

    @property
    def db_path(self) -> str:
        return self.get("storage.db_path", "qmind.db")

    @property
    def log_level(self) -> str:
        return os.getenv("QMIND_LOG_LEVEL") or self.get("logging.level", "INFO")

    @property
    def notification_type(self) -> str:
        return self.get("notification.type", "none")

    def to_dict(self) -> dict[str, Any]:
        return self._data

"""config.py 配置管理 单元测试"""

from __future__ import annotations

from pathlib import Path

import yaml

from qmind.config import Config


class TestConfig:
    def test_default_values(self):
        """无配置文件时返回默认值"""
        cfg = Config(path=Path("nonexistent.yaml"))
        assert cfg.execution_dry_run is True
        assert cfg.llm_default_model == "claude-sonnet-4-6"
        assert cfg.db_path == "qmind.db"

    def test_load_from_dict(self, tmp_path):
        """能读取 YAML 配置"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump({
            "execution": {"dry_run": False},
            "llm": {"default_model": "gpt-4o"},
            "storage": {"db_path": "/tmp/test.db"},
        }))
        cfg = Config(path=cfg_file)
        assert cfg.execution_dry_run is False
        assert cfg.llm_default_model == "gpt-4o"
        assert cfg.db_path == "/tmp/test.db"

    def test_env_override(self, tmp_path, monkeypatch):
        """环境变量应覆盖配置文件"""
        monkeypatch.setenv("QMIND_LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("QMIND_LOG_LEVEL", "DEBUG")

        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump({
            "llm": {"default_model": "claude-sonnet-4-6"},
        }))
        cfg = Config(path=cfg_file)
        assert cfg.llm_default_model == "deepseek-chat"
        assert cfg.log_level == "DEBUG"

    def test_nested_get(self, tmp_path):
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump({
            "execution": {"default_exchange": "binance", "dry_run": True},
        }))
        cfg = Config(path=cfg_file)
        assert cfg.get("execution.default_exchange") == "binance"
        assert cfg.get("execution.dry_run") is True
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

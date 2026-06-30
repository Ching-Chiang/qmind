"""main.py CLI 入口 单元测试 (Click CliRunner)"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qmind.main import cli


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_version(self, runner):
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "QMind" in result.output

    def test_analyze_no_args_shows_help(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "SYMBOL" in result.output

    def test_backtest_help(self, runner):
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output

    def test_watch_help(self, runner):
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "SYMBOLS" in result.output

    def test_learn_help(self, runner):
        result = runner.invoke(cli, ["learn", "--help"])
        assert result.exit_code == 0
        assert "--from-log" in result.output

    def test_list_strategies(self, runner):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "ma_cross" in result.output
        assert "macd" in result.output

    def test_price_help(self, runner):
        result = runner.invoke(cli, ["price", "--help"])
        assert result.exit_code == 0
        assert "SYMBOL" in result.output

    def test_backtest_unknown_strategy(self, runner):
        result = runner.invoke(cli, ["backtest", "--strategy", "nonexistent",
                                      "--start", "2024-01", "--end", "2024-06"])
        assert result.exit_code == 0  # CLI handles gracefully
        assert "未知策略" in result.output

    def test_price_no_args_fails(self, runner):
        result = runner.invoke(cli, ["price"])
        assert result.exit_code != 0  # missing argument

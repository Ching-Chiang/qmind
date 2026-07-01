"""execution/cex/okx.py & bybit.py 交易所适配器 单元测试 (mock ccxt)

覆盖范围:
  1. OKX: 构造器 + testnet 配置
  2. OKX: dry_run 标志行为
  3. OKX: ExchangeBase 继承
  4. Bybit: 构造器 + testnet 配置
  5. Bybit: dry_run 标志行为
  6. Bybit: ExchangeBase 继承
  7. 工厂创建 OKX / Bybit 实例
  8. 所有抽象方法实现校验
  9. ccxt rate limiter 配置验证

Mock 策略:
  ccxt 未安装在环境中，故通过往 sys.modules 注入 types.ModuleType 虚模块
  来拦截 ``import ccxt.async_support as ccxt``。虚模块只需要 __path__ 和
  __package__ 属性即可通过 Python 3.11+ import 机制校验。
"""

from __future__ import annotations

import asyncio
import sys
import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qmind.execution.base import ExchangeBase
from qmind.execution.cex.okx import OKXExchange
from qmind.execution.cex.bybit import BybitExchange
from qmind.execution.factory import ExchangeFactory


def _run(coro):
    """同步运行异步协程。"""
    return asyncio.run(coro)


# ===================================================================
# ccxt 虚模块工厂
# ===================================================================


def _make_ccxt_async_module() -> types.ModuleType:
    """创建一个 ccxt.async_support 虚模块。

    Python 的 ``import ccxt.async_support as ccxt`` 要求 ccxt 在
    sys.modules 中存在且可被识别为包（有 __path__），子模块也需可被
    getattr 访问。返回的模块对象可以挂载 ``okx``/``bybit`` 等属性。
    """
    mod = types.ModuleType("ccxt.async_support")
    mod.__package__ = "ccxt.async_support"
    mod.__path__ = ["/fake/ccxt/async_support"]
    return mod


def _make_ccxt_package() -> types.ModuleType:
    """创建一个 ccxt 顶层虚包，并将 async_support 子模块挂载为其属性。"""
    mod = types.ModuleType("ccxt")
    mod.__package__ = "ccxt"
    mod.__path__ = ["/fake/ccxt"]
    return mod


@contextmanager
def _patch_okx():
    """Mock ccxt 供 OKXExchange 测试。

    用法::

        with _patch_okx() as (mock_exchange, mock_okx_class):
            mock_exchange.fetch_ticker = AsyncMock(return_value={...})
            ex = OKXExchange(dry_run=False)
            ...

    Yields:
        (mock_exchange, mock_okx_class)
        - mock_exchange: ccxt.okx(config) 返回的交易所实例 mock
        - mock_okx_class: ``ccxt.okx`` 可调用对象 mock，用于验证构造参数
    """
    mock_exchange = MagicMock()
    mock_okx_class = MagicMock(return_value=mock_exchange)

    ccxt_pkg = _make_ccxt_package()
    async_mod = _make_ccxt_async_module()
    async_mod.okx = mock_okx_class
    ccxt_pkg.async_support = async_mod

    with patch.dict(sys.modules, {"ccxt": ccxt_pkg, "ccxt.async_support": async_mod}):
        yield mock_exchange, mock_okx_class


@contextmanager
def _patch_bybit():
    """Mock ccxt 供 BybitExchange 测试。

    用法与 _patch_okx 相同。Bybit 额外会调用 exchange.set_sandbox_mode()，
    mock_exchange 会自动响应该调用。

    Yields:
        (mock_exchange, mock_bybit_class)
    """
    mock_exchange = MagicMock()
    mock_bybit_class = MagicMock(return_value=mock_exchange)

    ccxt_pkg = _make_ccxt_package()
    async_mod = _make_ccxt_async_module()
    async_mod.bybit = mock_bybit_class
    ccxt_pkg.async_support = async_mod

    with patch.dict(sys.modules, {"ccxt": ccxt_pkg, "ccxt.async_support": async_mod}):
        yield mock_exchange, mock_bybit_class


# ===================================================================
# OKX
# ===================================================================


class TestOKXExchangeConstructor:
    """OKX 构造器 + testnet 配置"""

    def test_defaults(self):
        """默认参数：testnet=True, dry_run=True, 凭据为空。"""
        ex = OKXExchange()
        assert ex.name == "okx"
        assert ex.api_key == ""
        assert ex.api_secret == ""
        assert ex.passphrase == ""
        assert ex.testnet is True
        assert ex.dry_run is True
        assert ex._ccxt is None  # 懒加载未触发

    def test_custom_credentials(self):
        """传入完整凭据和 testnet=False。"""
        ex = OKXExchange(
            api_key="k123",
            api_secret="s456",
            passphrase="p789",
            testnet=False,
            dry_run=False,
        )
        assert ex.api_key == "k123"
        assert ex.api_secret == "s456"
        assert ex.passphrase == "p789"
        assert ex.testnet is False
        assert ex.dry_run is False

    def test_exchange_lazy_init_testnet(self):
        """testnet=True -> ccxt 获得测试网 URL 配置。"""
        with _patch_okx() as (mock_exchange, mock_okx_class):
            ex = OKXExchange(api_key="k", api_secret="s", passphrase="p", testnet=True)
            instance = ex.exchange

        assert instance is mock_exchange
        mock_okx_class.assert_called_once()
        config = mock_okx_class.call_args[0][0]

        assert config["apiKey"] == "k"
        assert config["secret"] == "s"
        assert config["password"] == "p"
        assert config["enableRateLimit"] is True
        assert config["options"]["defaultType"] == "swap"
        assert config["urls"]["api"]["rest"] == OKXExchange.REST_BASE_TESTNET

    def test_exchange_lazy_init_mainnet(self):
        """testnet=False -> ccxt 正常配置，无 urls 覆盖。"""
        with _patch_okx() as (mock_exchange, mock_okx_class):
            ex = OKXExchange(api_key="k", api_secret="s", passphrase="p", testnet=False, dry_run=False)
            _ = ex.exchange

        config = mock_okx_class.call_args[0][0]
        assert "urls" not in config

    def test_exchange_property_cached(self):
        """exchange 属性是懒加载且只初始化一次。"""
        with _patch_okx() as (mock_exchange, mock_okx_class):
            ex = OKXExchange(dry_run=False)
            a = ex.exchange
            b = ex.exchange

        assert a is b
        mock_okx_class.assert_called_once()

    def test_rate_limiter_enabled(self):
        """OKX 启用 ccxt 内置 rate limiter。"""
        with _patch_okx() as (mock_exchange, mock_okx_class):
            ex = OKXExchange(dry_run=False)
            _ = ex.exchange

        config = mock_okx_class.call_args[0][0]
        assert config["enableRateLimit"] is True


class TestOKXExchangeDryRun:
    """OKX dry_run 标志行为"""

    def test_dry_run_place_order(self):
        """dry_run=True -> place_order 返回模拟填充订单。"""
        ex = OKXExchange(dry_run=True)
        result = _run(ex.place_order("BTC/USDT", "buy", "limit", 0.15, 87000.0))

        assert result.order_id.startswith("dry_")
        assert result.symbol == "BTC/USDT"
        assert result.side == "buy"
        assert result.type == "limit"
        assert result.price == 87000.0
        assert result.quantity == 0.15
        assert result.status == "filled"
        assert result.filled_quantity == 0.15
        assert result.avg_fill_price == 87000.0

    def test_dry_run_flag_accessible(self):
        """dry_run=False 时标志正确设置。"""
        ex = OKXExchange(dry_run=False)
        assert ex.dry_run is False

    def test_dry_run_default_is_true(self):
        """OKXExchange 默认 dry_run=True（安全默认）。"""
        ex = OKXExchange()
        assert ex.dry_run is True


class TestOKXExchangeBase:
    """OKX 继承与抽象方法实现"""

    def test_inherits_exchange_base(self):
        assert issubclass(OKXExchange, ExchangeBase)

    def test_implements_all_abstract(self):
        """OKXExchange 实现了 ExchangeBase 的所有抽象方法。"""
        for m in ("get_price", "get_balance", "place_order", "cancel_order", "get_order", "get_positions"):
            assert hasattr(OKXExchange, m), f"缺少抽象方法: {m}"
            assert callable(getattr(OKXExchange, m)), f"{m} 不可调用"

    def test_has_close_method(self):
        """提供 close() 生命周期方法。"""
        assert hasattr(OKXExchange, "close") and callable(OKXExchange.close)


class TestOKXExchangeAPIMocked:
    """OKX API 方法（ccxt mock）"""

    def test_get_price(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
            ex = OKXExchange(dry_run=False)
            price = _run(ex.get_price("BTC/USDT"))

        assert price == 50000.0
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    def test_get_balance(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_balance = AsyncMock(return_value={
                "total": {"USDT": 10000, "BTC": 0.5},
                "free": {"USDT": 8000, "BTC": 0.3},
                "used": {"USDT": 2000, "BTC": 0.2},
            })
            ex = OKXExchange(dry_run=False)
            balances = _run(ex.get_balance())

        assert len(balances) == 2
        assert balances[0].asset == "USDT"
        assert balances[0].total == 10000
        assert balances[0].free == 8000
        assert balances[0].locked == 2000

    def test_place_order_live_limit(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.create_limit_order = AsyncMock(return_value={
                "id": "abc123", "symbol": "BTC/USDT", "side": "buy",
                "type": "limit", "price": 87000.0, "amount": 0.15,
                "status": "open", "filled": 0.0, "average": None,
            })
            ex = OKXExchange(dry_run=False)
            result = _run(ex.place_order("BTC/USDT", "buy", "limit", 0.15, 87000.0))

        assert result.order_id == "abc123"
        assert result.status == "open"
        mock_exchange.create_limit_order.assert_called_once_with(
            "BTC/USDT", "buy", 0.15, 87000.0, params={}
        )

    def test_place_order_live_market(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.create_market_order = AsyncMock(return_value={
                "id": "mkt1", "symbol": "BTC/USDT", "side": "sell",
                "type": "market", "price": None, "amount": 0.1,
                "status": "filled", "filled": 0.1, "average": 49800.0,
            })
            ex = OKXExchange(dry_run=False)
            result = _run(ex.place_order("BTC/USDT", "sell", "market", 0.1))

        assert result.order_id == "mkt1"
        assert result.status == "filled"
        mock_exchange.create_market_order.assert_called_once_with(
            "BTC/USDT", "sell", 0.1, params={}
        )

    def test_cancel_order_success(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.cancel_order = AsyncMock(return_value={"id": "abc"})
            ex = OKXExchange(dry_run=False)
            result = _run(ex.cancel_order("BTC/USDT", "abc"))

        assert result is True
        mock_exchange.cancel_order.assert_called_once_with("abc", "BTC/USDT")

    def test_cancel_order_failure_returns_false(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.cancel_order = AsyncMock(side_effect=Exception("Network error"))
            ex = OKXExchange(dry_run=False)
            result = _run(ex.cancel_order("BTC/USDT", "abc"))

        assert result is False

    def test_get_order(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_order = AsyncMock(return_value={
                "id": "abc", "symbol": "BTC/USDT", "side": "buy",
                "type": "limit", "price": 87000.0, "amount": 0.15,
                "status": "filled", "filled": 0.15, "average": 87100.0,
            })
            ex = OKXExchange(dry_run=False)
            result = _run(ex.get_order("BTC/USDT", "abc"))

        assert result is not None
        assert result.order_id == "abc"
        assert result.status == "filled"
        assert result.avg_fill_price == 87100.0
        mock_exchange.fetch_order.assert_called_once_with("abc", "BTC/USDT")

    def test_get_order_not_found(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_order = AsyncMock(side_effect=Exception("Order not found"))
            ex = OKXExchange(dry_run=False)
            result = _run(ex.get_order("BTC/USDT", "nonexistent"))

        assert result is None

    def test_get_positions(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.5,
                    "entryPrice": 85000.0,
                    "markPrice": 87000.0,
                    "unrealizedPnl": 1000.0,
                    "leverage": 5,
                }
            ])
            ex = OKXExchange(dry_run=False)
            positions = _run(ex.get_positions())

        assert len(positions) == 1
        p = positions[0]
        assert p.symbol == "BTC/USDT"
        assert p.side == "long"
        assert p.quantity == 0.5
        assert p.entry_price == 85000.0
        assert p.pnl_unrealized == 1000.0
        assert p.leverage == 5

    def test_get_positions_empty(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[])
            ex = OKXExchange(dry_run=False)
            positions = _run(ex.get_positions("ETH/USDT"))

        assert positions == []

    def test_get_positions_skips_zero_contracts(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {"symbol": "BTC/USDT", "contracts": 0, "side": "long",
                 "entryPrice": 0, "markPrice": 0, "unrealizedPnl": 0, "leverage": 1},
                {"symbol": "ETH/USDT", "contracts": 2.0, "side": "long",
                 "entryPrice": 3000, "markPrice": 3100, "unrealizedPnl": 200, "leverage": 3},
            ])
            ex = OKXExchange(dry_run=False)
            positions = _run(ex.get_positions())

        assert len(positions) == 1
        assert positions[0].symbol == "ETH/USDT"

    def test_close(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.close = AsyncMock()
            ex = OKXExchange(dry_run=False)
            _ = ex.exchange
            _run(ex.close())

        mock_exchange.close.assert_called_once()

    def test_close_sets_ccxt_to_none(self):
        with _patch_okx() as (mock_exchange, _):
            mock_exchange.close = AsyncMock()
            ex = OKXExchange(dry_run=False)
            _ = ex.exchange
            _run(ex.close())

        assert ex._ccxt is None


# ===================================================================
# Bybit
# ===================================================================


class TestBybitExchangeConstructor:
    """Bybit 构造器 + testnet 配置"""

    def test_defaults(self):
        """默认参数：testnet=True, dry_run=True, market_type='linear'。"""
        ex = BybitExchange()
        assert ex.name == "bybit"
        assert ex.api_key == ""
        assert ex.api_secret == ""
        assert ex.testnet is True
        assert ex.dry_run is True
        assert ex.market_type == "linear"
        assert ex._ccxt is None

    def test_custom_credentials(self):
        """传入完整凭据、testnet=False, market_type='spot'。"""
        ex = BybitExchange(
            api_key="k123",
            api_secret="s456",
            testnet=False,
            dry_run=False,
            market_type="spot",
        )
        assert ex.api_key == "k123"
        assert ex.api_secret == "s456"
        assert ex.testnet is False
        assert ex.dry_run is False
        assert ex.market_type == "spot"

    def test_exchange_lazy_init_linear(self):
        """market_type='linear' -> ccxt 配置 defaultType='linear' + sandbox。"""
        with _patch_bybit() as (mock_exchange, mock_bybit_class):
            ex = BybitExchange(api_key="k", api_secret="s", testnet=True)
            instance = ex.exchange

        assert instance is mock_exchange
        mock_bybit_class.assert_called_once()
        config = mock_bybit_class.call_args[0][0]
        assert config["apiKey"] == "k"
        assert config["secret"] == "s"
        assert config["enableRateLimit"] is True
        assert config["rateLimit"] == 50
        assert config["options"]["defaultType"] == "linear"
        mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_exchange_lazy_init_spot(self):
        """market_type='spot' + testnet=False -> no sandbox, defaultType='spot'。"""
        with _patch_bybit() as (mock_exchange, mock_bybit_class):
            ex = BybitExchange(market_type="spot", testnet=False, dry_run=False)
            _ = ex.exchange

        config = mock_bybit_class.call_args[0][0]
        assert config["options"]["defaultType"] == "spot"
        mock_exchange.set_sandbox_mode.assert_not_called()

    def test_exchange_lazy_init_inverse(self):
        """market_type='inverse' -> defaultType='inverse'。"""
        with _patch_bybit() as (mock_exchange, mock_bybit_class):
            ex = BybitExchange(market_type="inverse", dry_run=False)
            _ = ex.exchange

        config = mock_bybit_class.call_args[0][0]
        assert config["options"]["defaultType"] == "inverse"

    def test_exchange_property_cached(self):
        """exchange 属性是懒加载且只初始化一次。"""
        with _patch_bybit() as (mock_exchange, mock_bybit_class):
            ex = BybitExchange(dry_run=False)
            a = ex.exchange
            b = ex.exchange

        assert a is b
        mock_bybit_class.assert_called_once()

    def test_rate_limiter_config(self):
        """Bybit 启用 rate limiter 并设 50ms 间隔。"""
        with _patch_bybit() as (mock_exchange, mock_bybit_class):
            ex = BybitExchange(dry_run=False)
            _ = ex.exchange

        config = mock_bybit_class.call_args[0][0]
        assert config["enableRateLimit"] is True
        assert config["rateLimit"] == 50


class TestBybitExchangeDryRun:
    """Bybit dry_run 标志行为"""

    def test_dry_run_place_order_limit(self):
        """dry_run=True -> limit 订单返回模拟填充。"""
        ex = BybitExchange(dry_run=True)
        result = _run(ex.place_order("BTC/USDT", "buy", "limit", 0.15, 87000.0))

        assert result.order_id.startswith("dry_")
        assert "bybit" in result.order_id
        assert result.symbol == "BTC/USDT"
        assert result.side == "buy"
        assert result.type == "limit"
        assert result.price == 87000.0
        assert result.quantity == 0.15
        assert result.status == "filled"
        assert result.filled_quantity == 0.15
        assert result.avg_fill_price == 87000.0

    def test_dry_run_place_order_market(self):
        """dry_run=True -> 市价单 avg_fill_price 为 0（无价格参考）。"""
        ex = BybitExchange(dry_run=True)
        result = _run(ex.place_order("BTC/USDT", "sell", "market", 0.1))

        assert result.type == "market"
        assert result.price == 0.0  # market 单 price 无意义
        assert result.status == "filled"

    def test_dry_run_default_is_true(self):
        assert BybitExchange().dry_run is True

    def test_dry_run_flag_false(self):
        assert BybitExchange(dry_run=False).dry_run is False


class TestBybitExchangeBase:
    """Bybit 继承与抽象方法实现"""

    def test_inherits_exchange_base(self):
        assert issubclass(BybitExchange, ExchangeBase)

    def test_implements_all_abstract(self):
        """BybitExchange 实现了 ExchangeBase 的所有抽象方法。"""
        for m in ("get_price", "get_balance", "place_order", "cancel_order", "get_order", "get_positions"):
            assert hasattr(BybitExchange, m), f"缺少抽象方法: {m}"
            assert callable(getattr(BybitExchange, m)), f"{m} 不可调用"

    def test_has_close_method(self):
        assert hasattr(BybitExchange, "close") and callable(BybitExchange.close)


class TestBybitExchangeAPIMocked:
    """Bybit API 方法（ccxt mock）"""

    def test_get_price(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 52000.0})
            ex = BybitExchange(dry_run=False)
            price = _run(ex.get_price("BTC/USDT"))

        assert price == 52000.0
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    def test_get_balance(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_balance = AsyncMock(return_value={
                "total": {"USDT": 25000, "ETH": 10},
                "free": {"USDT": 20000, "ETH": 8},
                "used": {"USDT": 5000, "ETH": 2},
            })
            ex = BybitExchange(dry_run=False)
            balances = _run(ex.get_balance())

        assert len(balances) == 2
        assert balances[1].asset == "ETH"
        assert balances[1].total == 10

    def test_place_order_live_limit(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.create_limit_order = AsyncMock(return_value={
                "id": "bybit123", "symbol": "BTC/USDT", "side": "buy",
                "type": "limit", "price": 86000.0, "amount": 0.2,
                "status": "open", "filled": 0.0, "average": None,
            })
            ex = BybitExchange(dry_run=False)
            result = _run(ex.place_order("BTC/USDT", "buy", "limit", 0.2, 86000.0))

        assert result.order_id == "bybit123"
        assert result.status == "open"
        mock_exchange.create_limit_order.assert_called_once_with(
            "BTC/USDT", "buy", 0.2, 86000.0, {}
        )

    def test_place_order_live_market(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.create_market_order = AsyncMock(return_value={
                "id": "mkt_bybit", "symbol": "BTC/USDT", "side": "sell",
                "type": "market", "price": None, "amount": 0.1,
                "status": "filled", "filled": 0.1, "average": 51800.0,
            })
            ex = BybitExchange(dry_run=False)
            result = _run(ex.place_order("BTC/USDT", "sell", "market", 0.1))

        assert result.status == "filled"
        assert result.avg_fill_price == 51800.0
        mock_exchange.create_market_order.assert_called_once_with(
            "BTC/USDT", "sell", 0.1, {}
        )

    def test_place_order_limit_price_zero_raises(self):
        """Bybit 限价单 price=0 抛出 ValueError (不会走到 ccxt)。"""
        with _patch_bybit() as (mock_exchange, _):
            ex = BybitExchange(dry_run=False)
            with pytest.raises(ValueError, match="Limit orders require a valid price > 0"):
                _run(ex.place_order("BTC/USDT", "buy", "limit", 0.1, price=0.0))

    def test_cancel_order_success(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.cancel_order = AsyncMock(return_value={"id": "abc"})
            ex = BybitExchange(dry_run=False)
            result = _run(ex.cancel_order("BTC/USDT", "abc"))

        assert result is True
        mock_exchange.cancel_order.assert_called_once_with("abc", "BTC/USDT")

    def test_cancel_order_failure_returns_false(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.cancel_order = AsyncMock(side_effect=Exception("err"))
            ex = BybitExchange(dry_run=False)
            result = _run(ex.cancel_order("BTC/USDT", "abc"))

        assert result is False

    def test_get_order(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_order = AsyncMock(return_value={
                "id": "abc", "symbol": "BTC/USDT", "side": "sell",
                "type": "market", "price": 0.0, "amount": 0.5,
                "status": "filled", "filled": 0.5, "average": 49000.0,
            })
            ex = BybitExchange(dry_run=False)
            result = _run(ex.get_order("BTC/USDT", "abc"))

        assert result is not None
        assert result.order_id == "abc"
        assert result.avg_fill_price == 49000.0
        mock_exchange.fetch_order.assert_called_once_with("abc", "BTC/USDT")

    def test_get_order_not_found(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_order = AsyncMock(side_effect=Exception("not found"))
            ex = BybitExchange(dry_run=False)
            result = _run(ex.get_order("BTC/USDT", "bad_id"))

        assert result is None

    def test_get_positions(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {
                    "symbol": "BTC/USDT",
                    "contracts": 1.0,
                    "entryPrice": 84000.0,
                    "markPrice": 86000.0,
                    "unrealizedPnl": 2000.0,
                    "leverage": 10,
                }
            ])
            ex = BybitExchange(dry_run=False)
            positions = _run(ex.get_positions())

        assert len(positions) == 1
        p = positions[0]
        assert p.symbol == "BTC/USDT"
        assert p.side == "long"  # contracts=1.0 > 0 -> long
        assert p.quantity == 1.0
        assert p.entry_price == 84000.0
        assert p.pnl_unrealized == 2000.0
        assert p.leverage == 10

    def test_get_positions_short(self):
        """负数 contracts -> side='short'。"""
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {"symbol": "ETH/USDT", "contracts": -2.5,
                 "entryPrice": 3200, "markPrice": 3100,
                 "unrealizedPnl": 250, "leverage": 3}
            ])
            ex = BybitExchange(dry_run=False)
            positions = _run(ex.get_positions("ETH/USDT"))

        assert len(positions) == 1
        assert positions[0].side == "short"
        assert positions[0].quantity == 2.5

    def test_get_positions_skips_zero_contracts(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {"symbol": "BTC/USDT", "contracts": 0,
                 "entryPrice": 0, "markPrice": 0,
                 "unrealizedPnl": 0, "leverage": 1},
            ])
            ex = BybitExchange(dry_run=False)
            positions = _run(ex.get_positions())

        assert positions == []

    def test_close(self):
        with _patch_bybit() as (mock_exchange, _):
            mock_exchange.close = AsyncMock()
            ex = BybitExchange(dry_run=False)
            _ = ex.exchange
            _run(ex.close())

        mock_exchange.close.assert_called_once()
        assert ex._ccxt is None


# ===================================================================
# 工厂集成
# ===================================================================


class TestExchangeFactoryCEX:
    """工厂创建 OKX / Bybit 实例"""

    def test_factory_returns_dry_run_when_flag_true(self):
        """dry_run=True 时无论 name 是什么都返回 DryRunExchange。"""
        ex = ExchangeFactory.create("okx", dry_run=True)
        assert ex.name == "dry_run"

        ex = ExchangeFactory.create("bybit", dry_run=True)
        assert ex.name == "dry_run"

    def test_factory_creates_okx(self):
        """dry_run=False, name='okx' -> OKXExchange。"""
        ex = ExchangeFactory.create("okx", dry_run=False, config={
            "api_key": "k", "api_secret": "s", "passphrase": "p", "testnet": True,
        })
        assert isinstance(ex, OKXExchange)
        assert ex.name == "okx"
        assert ex.api_key == "k"
        assert ex.api_secret == "s"
        assert ex.passphrase == "p"
        assert ex.testnet is True
        assert ex.dry_run is False

    def test_factory_creates_bybit(self):
        """dry_run=False, name='bybit' -> BybitExchange（testnet 默认为 True）。"""
        ex = ExchangeFactory.create("bybit", dry_run=False, config={
            "api_key": "k", "api_secret": "s",
        })
        assert isinstance(ex, BybitExchange)
        assert ex.name == "bybit"
        assert ex.api_key == "k"
        assert ex.api_secret == "s"
        assert ex.testnet is True  # factory 未透传 testnet，保持类默认 True
        assert ex.dry_run is False
        assert ex.market_type == "linear"

    def test_factory_okx_passes_passphrase(self):
        """OKX 特有的 passphrase 从 config 透传。"""
        ex = ExchangeFactory.create("okx", dry_run=False, config={
            "passphrase": "mysecret", "api_key": "k", "api_secret": "s",
        })
        assert ex.passphrase == "mysecret"

    def test_factory_okx_defaults_to_testnet(self):
        """OKX 默认 testnet=True。"""
        ex = ExchangeFactory.create("okx", dry_run=False, config={
            "api_key": "k", "api_secret": "s",
        })
        assert ex.testnet is True

    def test_factory_bybit_defaults_to_testnet(self):
        """Bybit 默认 testnet=True。"""
        ex = ExchangeFactory.create("bybit", dry_run=False, config={
            "api_key": "k", "api_secret": "s",
        })
        assert ex.testnet is True

    def test_factory_bybit_ignores_unknown_config_keys(self):
        """Bybit 忽略未知 config 键（不含 passphrase）。"""
        ex = ExchangeFactory.create("bybit", dry_run=False, config={
            "api_key": "k", "api_secret": "s", "passphrase": "ignored",
        })
        assert not hasattr(ex, "passphrase") or getattr(ex, "passphrase", "") == ""

"""data/symbol_map.py 符号格式转换 单元测试"""

from __future__ import annotations

from qmind.data.symbol_map import detect_source, to_binance, to_yfinance


class TestToYFinance:
    def test_crypto_slash(self):
        assert to_yfinance("BTC/USDT") == "BTC-USD"

    def test_crypto_dash(self):
        assert to_yfinance("ETH-USD") == "ETH-USD"

    def test_usdt_to_usd(self):
        assert to_yfinance("SOL/USDT") == "SOL-USD"

    def test_plain_ticker(self):
        assert to_yfinance("AAPL") == "AAPL"


class TestToBinance:
    def test_already_has_slash(self):
        assert to_binance("BTC/USDT") == "BTC/USDT"

    def test_without_slash(self):
        assert to_binance("BTCUSDT") == "BTC/USDT"

    def test_eth(self):
        assert to_binance("ETHUSDT") == "ETH/USDT"


class TestDetectSource:
    def test_crypto_slash(self):
        assert detect_source("BTC/USDT") == "binance"

    def test_crypto_dash(self):
        assert detect_source("ETH-USD") == "binance"

    def test_a_share_number(self):
        assert detect_source("000001") == "tushare"

    def test_a_share_sz(self):
        assert detect_source("000001.SZ") == "tushare"

    def test_a_share_sh(self):
        assert detect_source("600000.SH") == "tushare"

    def test_us_stock_default_yfinance(self):
        assert detect_source("AAPL") == "yfinance"

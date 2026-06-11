"""Tests for Kelly position sizing."""

import pytest

from src.backtest.sizing import KellySizer


@pytest.fixture
def sizer() -> KellySizer:
    return KellySizer(kelly_fraction=0.25, max_position_size=0.05)


def test_kelly_positive_edge(sizer: KellySizer):
    f = sizer.kelly_fraction(model_prob=0.55, market_price=0.40)
    assert f > 0
    assert f <= sizer.max_position_size


def test_kelly_never_exceeds_max_position(sizer: KellySizer):
    f = sizer.kelly_fraction(model_prob=0.99, market_price=0.01)
    assert f <= sizer.max_position_size


def test_kelly_zero_edge(sizer: KellySizer):
    f = sizer.kelly_fraction(model_prob=0.50, market_price=0.50)
    assert f == 0.0


def test_kelly_invalid_price(sizer: KellySizer):
    assert sizer.kelly_fraction(0.6, 0.0) == 0.0
    assert sizer.kelly_fraction(0.6, 1.0) == 0.0


def test_validate_sizing_clamps(sizer: KellySizer):
    assert sizer.validate_sizing(0.99) == 0.05
    assert sizer.validate_sizing(-0.1) == 0.0


def test_contracts_from_capital(sizer: KellySizer):
    n = sizer.contracts_from_capital(capital=10_000, fraction=0.05, price=0.50)
    assert n == 1000


def test_kelly_sell(sizer: KellySizer):
    f = sizer.kelly_fraction_sell(model_prob=0.30, market_price=0.55)
    assert f >= 0
    assert f <= sizer.max_position_size

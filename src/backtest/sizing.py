"""Kelly criterion position sizing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KellySizer:
    """
    Fractional Kelly position sizing for binary prediction market contracts.

    f* = (bp - q) / b  where b = net odds, p = model prob, q = 1 - p
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_size: float = 0.05,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size

    def kelly_fraction_raw(
        self,
        model_prob: float,
        market_price: float,
    ) -> float:
        """
        Compute full Kelly fraction before scaling and caps.

        market_price is the executable price (ask when buying YES).
        model_prob is estimated true probability of YES (home win).
        """
        p = model_prob
        q = 1.0 - p
        if market_price <= 0 or market_price >= 1:
            return 0.0
        b = (1.0 - market_price) / market_price  # net odds on YES
        if b <= 0:
            return 0.0
        f_star = (b * p - q) / b
        return max(0.0, f_star)

    def kelly_fraction(
        self,
        model_prob: float,
        market_price: float,
        kelly_fraction: float | None = None,
    ) -> float:
        """Compute fractional Kelly fraction of capital to deploy."""
        frac = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        f_star = self.kelly_fraction_raw(model_prob, market_price)
        return self.validate_sizing(f_star * frac)

    def kelly_fraction_sell(
        self,
        model_prob: float,
        market_price: float,
        kelly_fraction: float | None = None,
    ) -> float:
        """Kelly fraction for selling YES (fading overpriced home win)."""
        frac = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        # Selling YES ≡ buying NO at (1 - bid); use complement symmetry
        f_star = self.kelly_fraction_raw(1.0 - model_prob, 1.0 - market_price)
        return self.validate_sizing(f_star * frac)

    def validate_sizing(self, fraction: float) -> float:
        """Clamp Kelly output to [0, max_position_size]."""
        return float(max(0.0, min(fraction, self.max_position_size)))

    def contracts_from_capital(
        self,
        capital: float,
        fraction: float,
        price: float,
    ) -> int:
        """Convert capital fraction to whole contracts at given price."""
        if price <= 0 or fraction <= 0:
            return 0
        dollars = capital * fraction
        return max(0, int(dollars / price))

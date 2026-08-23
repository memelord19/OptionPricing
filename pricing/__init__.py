"""
pricing
=======

A small option-pricing toolkit with three interchangeable pricing engines:

- ``binomial_tree``  - recombining CRR binomial tree (European & American)
- ``monte_carlo``    - Monte Carlo simulation under the risk-neutral measure
- ``black_scholes``  - closed-form Black-Scholes price, Greeks, implied vol

Each module can be imported directly, e.g.::

    from pricing.binomial_tree import PriceOptions, get_up_down_proba
    from pricing.monte_carlo import get_option_price_MC
    from pricing.black_scholes import BlackScholes

or the most common functions are re-exported here for convenience::

    from pricing import PriceOptions, get_option_price_MC, BlackScholes
"""

from .binomial_tree import (
    get_up_down_proba,
    PriceOptions,
    calculate_Greeks,
)
from .monte_carlo import (
    get_brownian_motion_samples,
    get_geometric_brownian_motion,
    get_option_price_MC,
    get_delta_MC,
    get_gamma_MC,
    get_vega_MC,
    get_rho_MC,
)
from .black_scholes import (
    BlackScholes,
    get_delta_BS,
    get_gamma_BS,
    get_vega_BS,
    get_rho_BS,
    GetImpliedVolatility,
)

__all__ = [
    # binomial_tree
    "get_up_down_proba",
    "PriceOptions",
    "calculate_Greeks",
    # monte_carlo
    "get_brownian_motion_samples",
    "get_geometric_brownian_motion",
    "get_option_price_MC",
    "get_delta_MC",
    "get_gamma_MC",
    "get_vega_MC",
    "get_rho_MC",
    # black_scholes
    "BlackScholes",
    "get_delta_BS",
    "get_gamma_BS",
    "get_vega_BS",
    "get_rho_BS",
    "GetImpliedVolatility",
]

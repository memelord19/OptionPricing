"""
monte_carlo.py
===============

Monte Carlo option pricer under the risk-neutral measure: simulate geometric
Brownian motion paths, average discounted payoffs, and estimate Greeks by
bump-and-reprice with common random numbers.
"""

import numpy as np


def get_brownian_motion_samples(n: int, volatility: float, T: float, M: int) -> np.ndarray:
    """
    Simulate M sample paths of a standard Brownian motion W_t over [0, T]
    using n equal time steps.

    Parameters
    ----------
    n : int
        Number of time steps.
    volatility : float
        Included for signature symmetry with get_geometric_brownian_motion;
        not used here since standard Brownian motion has unit diffusion.
    T : float
        Total horizon, in years.
    M : int
        Number of simulated paths.

    Returns
    -------
    np.ndarray, shape (n+1, M)
        Each column is one simulated path, row 0 is t=0 (all zeros).
    """
    dt = T / n
    # Standard deviation of a N(0, dt) increment is sqrt(dt), not dt.
    increments = np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    origins = np.zeros((1, M))
    samples = np.vstack([origins, increments]).cumsum(axis=0)
    return samples


def get_geometric_brownian_motion(n: int, volatility: float, T: float, M: int, mu: float) -> np.ndarray:
    """
    Simulate M sample paths of a geometric Brownian motion, normalized to
    start at 1.0 (multiply by a spot price to get actual stock-price paths).

    Parameters
    ----------
    n : int
        Number of time steps.
    volatility : float
        Annualized volatility (sigma).
    T : float
        Total horizon, in years.
    M : int
        Number of simulated paths.
    mu : float
        Drift. For risk-neutral pricing, pass the risk-free rate r here.

    Returns
    -------
    np.ndarray, shape (n+1, M)
        Each column is one simulated path of S_t / S_0, row 0 is t=0 (all ones).
    """
    dt = T / n
    # Standard deviation of each step's Gaussian shock is sqrt(dt), not dt.
    z = np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    step_returns = np.exp((mu - volatility ** 2 / 2) * dt + volatility * z)
    StockPrices = np.vstack([np.ones((1, M)), step_returns]).cumprod(axis=0)
    return StockPrices


def get_option_price_MC(riskFree: float, sigma: float, Strike: float, Spot: float,
                         n: int, M: int, T: float, call: int = 1) -> tuple:
    """
    Price a European option by Monte Carlo simulation under the risk-neutral
    measure (drift = riskFree).

    Parameters
    ----------
    riskFree : float
        Annualized, continuously-compounded risk-free rate.
    sigma : float
        Annualized volatility of the underlying.
    Strike : float
        Strike price.
    Spot : float
        Current stock price.
    n : int
        Number of time steps used to build each simulated path.
    M : int
        Number of simulated paths.
    T : float
        Time to maturity, in years.
    call : int, optional
        1 for a call, -1 for a put (default 1), matching PriceOptions's convention.

    Returns
    -------
    (price, stderr) : tuple of float
        price  - discounted Monte Carlo estimate of the option price.
        stderr - standard error of that estimate (discounted payoff std / sqrt(M)),
                 a rough 1-sigma measure of how much to trust `price`.
    """
    StockPrices = Spot * get_geometric_brownian_motion(n, sigma, T, M, riskFree)
    terminal = StockPrices[-1]
    payoffs = np.maximum(call * (terminal - Strike), 0)
    discounted = np.exp(-riskFree * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(M)
    return price, stderr


def get_delta_MC(riskFree: float, sigma: float, Strike: float, Spot: float,
                  n: int, M: int, T: float, call: int = 1) -> float:
    """Central finite-difference estimate of delta = dPrice/dSpot."""
    deltaS = 0.01 * Spot  # bump proportional to spot for numerical stability
    np.random.seed(1)
    price_up, _ = get_option_price_MC(riskFree, sigma, Strike, Spot + deltaS, n, M, T, call)
    np.random.seed(1)  # common random numbers: same paths, only Spot differs
    price_down, _ = get_option_price_MC(riskFree, sigma, Strike, Spot - deltaS, n, M, T, call)
    return (price_up - price_down) / (2 * deltaS)


def get_gamma_MC(riskFree: float, sigma: float, Strike: float, Spot: float,
                  n: int, M: int, T: float, call: int = 1) -> float:
    """Central finite-difference estimate of gamma = d^2 Price / dSpot^2."""
    deltaS = 0.01 * Spot
    np.random.seed(1)
    price_up, _ = get_option_price_MC(riskFree, sigma, Strike, Spot + deltaS, n, M, T, call)
    np.random.seed(1)
    price_mid, _ = get_option_price_MC(riskFree, sigma, Strike, Spot, n, M, T, call)
    np.random.seed(1)
    price_down, _ = get_option_price_MC(riskFree, sigma, Strike, Spot - deltaS, n, M, T, call)
    return (price_up - 2 * price_mid + price_down) / (deltaS ** 2)


def get_vega_MC(riskFree: float, sigma: float, Strike: float, Spot: float,
                 n: int, M: int, T: float, call: int = 1) -> float:
    """Forward finite-difference estimate of vega = dPrice/dSigma."""
    deltaVol = 0.01
    np.random.seed(2)
    price_up, _ = get_option_price_MC(riskFree, sigma + deltaVol, Strike, Spot, n, M, T, call)
    np.random.seed(2)
    price_base, _ = get_option_price_MC(riskFree, sigma, Strike, Spot, n, M, T, call)
    return (price_up - price_base) / deltaVol


def get_rho_MC(riskFree: float, sigma: float, Strike: float, Spot: float,
               n: int, M: int, T: float, call: int = 1) -> float:
    """Forward finite-difference estimate of rho = dPrice/dRiskFree."""
    deltaR = 0.01
    np.random.seed(3)
    price_up, _ = get_option_price_MC(riskFree + deltaR, sigma, Strike, Spot, n, M, T, call)
    np.random.seed(3)
    price_base, _ = get_option_price_MC(riskFree, sigma, Strike, Spot, n, M, T, call)
    return (price_up - price_base) / deltaR

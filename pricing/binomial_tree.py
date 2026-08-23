"""
binomial_tree.py
=================

Recombining Cox-Ross-Rubinstein (CRR) binomial tree pricer for European and
American vanilla options, plus finite-difference Greeks estimated from the
tree.
"""

import math

import numpy as np


def get_up_down_proba(volatility: float, period: float, riskFree: float):
    """
    Compute the CRR up/down multipliers and risk-neutral up-probability
    for a single time step.

    Parameters
    ----------
    volatility : float
        Annualized volatility of the underlying (sigma).
    period : float
        Length of ONE tree step, in years (i.e. T / n_steps, not total T).
    riskFree : float
        Annualized, continuously-compounded risk-free rate.

    Returns
    -------
    (up, down, p) : tuple of float
        up   - multiplicative up factor for one step
        down - multiplicative down factor for one step
        p    - risk-neutral probability of an up move
    """
    up = math.exp(volatility * math.sqrt(period))
    down = math.exp(-volatility * math.sqrt(period))
    p = (math.exp(riskFree * period) - down) / (up - down)
    return up, down, p


def PriceOptions(prices, step: int, position: int, n: int, stock: float, strike: float,
                  up: float, down: float, p: float, r: float, period: float,
                  call=1, american=0) -> float:
    """
    Price a vanilla option using a recombining binomial (CRR) tree.

    Parameters
    ----------
    prices : 2D array
        Memoization table holding the option value at each tree node.
    step : int
        Current time step in the tree.
    position : int
        Current position in the tree. Binomial trees are like random walks,
        so after k steps the position can range from -k to +k.
    n : int
        Total number of time steps in the tree.
    stock : float
        Price of the underlying asset at this node.
    strike : float
        Strike price of the option.
    up : float
        Multiplicative up factor per step (e.g. exp(sigma * sqrt(dt))).
    down : float
        Multiplicative down factor per step (e.g. exp(-sigma * sqrt(dt))).
    p : float
        Risk-neutral probability of an up move.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    period : float
        Length of ONE time step, in years.
    call : int, optional
        1 for a call option, -1 for a put option.
    american : int, optional
        0 for European, 1 for American (checks early exercise at each node).

    Returns
    -------
    float
        The fair value of the option at this node.
    """
    # Out-of-range guard (shouldn't trigger in normal recursion, kept for safety)
    if (step > n + 1) or (position + n > 2 * n + 1) or (position + n < 0) or (abs(position) > step):
        return 0

    # Memoized already? Return cached value.
    if prices[step][position + n] != -1:
        return prices[step][position + n]

    # Terminal payoff at maturity.
    if step == n:
        prices[step][position + n] = max(call * (stock - strike), 0)
        return prices[step][position + n]

    # Backward induction: discounted expectation of the two children.
    prices[step][position + n] = (
        p * PriceOptions(prices, step + 1, position + 1, n, stock * up, strike, up, down, p, r, period, call, american)
        + (1 - p) * PriceOptions(prices, step + 1, position - 1, n, stock * down, strike, up, down, p, r, period, call, american)
    )
    prices[step][position + n] *= math.exp(-r * period)

    # American early-exercise check.
    if american:
        intrinsic = max(call * (stock - strike), 0)
        prices[step][position + n] = max(prices[step][position + n], intrinsic)

    return prices[step][position + n]


def calculate_Greeks(tree_array, params: dict) -> dict:
    """
    Estimate delta, gamma, vega, and rho from a fully-populated CRR tree.

    Parameters
    ----------
    tree_array : 2D array
        The memoized `prices` table returned by PriceOptions, fully
        populated (i.e. after pricing the option at node (0, 0)).
        Must have been built with up/down derived from `params['Volatility']`
        via get_up_down_proba.
    params : dict
        Must contain: 'Volatility', 'Period' (TOTAL time to maturity, in
        years), 'StockPrice', 'StrikePrice', 'RiskFreeRate'.
        Optional: 'Call' (1 for call / -1 for put, default 1),
        'American' (0 or 1, default 0). These should match how the
        original tree was priced.

    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'rho'.

    Caveat
    ------
    This function assumes the tree it's handed was built with CRR
    up/down derived from `volatility` via get_up_down_proba. If a tree was
    built from hand-picked up/down values, the up/down this function
    re-derives from volatility won't match the tree's actual node spacing,
    and the Greeks will be wrong.
    """
    greeks = {}

    # Number of tree steps. tree_array has n_steps + 1 rows, so subtract 1.
    n_steps = tree_array.shape[0] - 1

    volatility = params['Volatility']
    T = params['Period']          # total time to maturity
    Stock = params['StockPrice']
    Strike = params['StrikePrice']
    r = params['RiskFreeRate']
    call = params.get('Call', 1)
    american = params.get('American', 0)
    dt = T / n_steps              # length of ONE step

    up, down, p = get_up_down_proba(volatility, dt, r)

    # --- Delta: slope of option value vs. stock price after one step ---
    delta = (tree_array[1][n_steps + 1] - tree_array[1][n_steps - 1]) / (Stock * (up - down))
    greeks['delta'] = delta

    # --- Gamma: curvature, using deltas estimated two steps out ---
    delta_up = (tree_array[2][n_steps + 2] - tree_array[2][n_steps]) / (Stock * (up**2 - down * up))
    delta_down = (tree_array[2][n_steps] - tree_array[2][n_steps - 2]) / (Stock * (up * down - down**2))
    gamma = (delta_up - delta_down) / (0.5 * Stock * (up**2 - down**2))
    greeks['gamma'] = gamma

    OptionPrice = tree_array[0][n_steps]

    # --- Vega: bump volatility by 1%, re-price a fresh tree ---
    deltaVol = 0.01
    up_v, down_v, p_v = get_up_down_proba(volatility + deltaVol, dt, r)
    prices_v = np.full((n_steps + 1, 2 * n_steps + 1), -1, dtype=np.float64)
    newPrice_v = PriceOptions(prices_v, 0, 0, n_steps, Stock, Strike, up_v, down_v, p_v, r, dt, call, american)
    greeks['vega'] = (newPrice_v - OptionPrice) / deltaVol

    # --- Rho: bump the risk-free rate by 1%, re-price a fresh tree ---
    # up/down don't depend on r, but p does - so p must be recomputed too.
    deltaR = 0.01
    up_r, down_r, p_r = get_up_down_proba(volatility, dt, r + deltaR)
    prices_r = np.full((n_steps + 1, 2 * n_steps + 1), -1, dtype=np.float64)
    newPrice_r = PriceOptions(prices_r, 0, 0, n_steps, Stock, Strike, up_r, down_r, p_r, r + deltaR, dt, call, american)
    greeks['rho'] = (newPrice_r - OptionPrice) / deltaR

    return greeks

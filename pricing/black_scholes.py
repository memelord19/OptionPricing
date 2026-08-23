"""
black_scholes.py
=================

Closed-form Black-Scholes pricing, Greeks, and Newton-Raphson implied
volatility for European options.
"""

import numpy as np
from scipy.stats import norm


def BlackScholes(S: float, sigma: float, K: float, T: float, r: float, type='C'):
    """
    Price a European option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current stock price.
    sigma : float
        Volatility.
    K : float
        Strike price.
    T : float
        Time to maturity, in years.
    r : float
        Risk-free interest rate.
    type : str, optional
        'C' for call, 'P' for put (default 'C').

    Returns
    -------
    float
        Option price, or None if an invalid type is given.
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        print('Invalid Type Given')
        return
    return price


def get_delta_BS(S: float, K: float, sigma: float, T: float, r: float, ty='C'):
    """
    Calculate the delta of a European option using Black-Scholes.

    Parameters
    ----------
    S, K, sigma, T, r : float
        Stock price, strike, volatility, time to maturity, risk-free rate.
    ty : str, optional
        'C' for call, 'P' for put (default 'C').
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if ty == 'C' else -norm.cdf(-d1)
    return delta


def get_gamma_BS(S: float, K: float, sigma: float, T: float, r: float, ty='C'):
    """
    Calculate the gamma of a European option using Black-Scholes.
    Gamma is the same for calls and puts.

    Parameters
    ----------
    S, K, sigma, T, r : float
        Stock price, strike, volatility, time to maturity, risk-free rate.
    ty : str, optional
        Unused (kept for a consistent signature with the other Greeks).
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    derD = 1 / (S * sigma * np.sqrt(T))
    gamma = norm.pdf(d1) * derD
    return gamma


def get_vega_BS(S: float, K: float, sigma: float, T: float, r: float):
    """
    Calculate the vega of a European option using Black-Scholes.
    Vega is the same for calls and puts.

    Parameters
    ----------
    S, K, sigma, T, r : float
        Stock price, strike, volatility, time to maturity, risk-free rate.
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    vega = np.sqrt(T) * S * norm.pdf(d1)
    return vega


def get_rho_BS(S: float, K: float, sigma: float, T: float, r: float, ty='C'):
    """
    Calculate the rho of a European option using Black-Scholes.

    Parameters
    ----------
    S, K, sigma, T, r : float
        Stock price, strike, volatility, time to maturity, risk-free rate.
    ty : str, optional
        'C' for call, 'P' for put (default 'C').
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if ty == 'C':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif ty == 'P':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        print('Invalid Type Given')
        return
    return rho


def GetImpliedVolatility(S, K, T: float, r, price, ty='C'):
    """
    Calculate the implied volatility of a European option using the
    Newton-Raphson method.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity, in years.
    r : float
        Risk-free interest rate.
    price : float
        Observed option price to invert.
    ty : str, optional
        'C' for call, 'P' for put (default 'C').

    Returns
    -------
    float
        Estimated implied volatility, or None if an invalid type is given.
    """
    sigma0 = 0.5
    tol = 0.001
    max_iter = 1000
    iter_count = 0
    sigma = 0.5
    if ty != 'C' and ty != 'P':
        print('Invalid Type Given')
        return
    while (iter_count < max_iter and abs(sigma0 - sigma) > tol) or iter_count == 0:
        price_calc = BlackScholes(S, sigma, K, T, r, ty)
        derC = get_vega_BS(S, K, sigma, T, r)
        sigma0 = sigma
        sigma = sigma0 - (price_calc - price) / derC
        iter_count += 1

    return sigma

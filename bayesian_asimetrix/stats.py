"""
Stats for Bayesian Posteriori Diagnostics
Developed by: Asimetrix, Data Science
Author: Ruben D. Vargas
"""

# Python Packages
import numpy as np
from scipy import stats
import pymc3 as pm

def mode_estimate(arr: np.array) -> float:
    """
    Estimates the mode of a numeric 1D-array.
    The mode of a numeric a numerica array is the point that maximizes the probability density function
    Args:
        arr : 1D-array, the data
    Returns:
        Distribution Mode
    """

    kde = stats.gaussian_kde(arr.T)
    pdfs = kde(arr.T)
    mode = arr[np.where(pdfs == pdfs.max())][0]

    return mode


def high_density_interval(arr: np.array, alpha: float=0.05) -> np.array:
    """
    Estimates the High Density Inteval from a numeric 1D-array.
    The high density interval are the values which are most credible and cover (1-alpha) of the whole distribution
    Args:
        arr : 1-D array, the data
        alpha: significance
    Returns:
        High Density Interval
    """
    return pm.hdi(arr, alpha)

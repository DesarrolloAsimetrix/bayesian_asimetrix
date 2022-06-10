"""
Stats for Bayesian Posteriori Diagnostics
Developed by: Asimetrix, Data Science
Author: Ruben D. Vargas
"""

# Python Packages
import numpy as np
import pandas as pd
from scipy import stats
import pymc3 as pm

def mode_estimate(arr: np.ndarray) -> float:
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

def high_density_interval(arr: np.ndarray, alpha: float=0.05) -> np.ndarray:
    """
    Estimates the High Density Inteval from a numeric 1D-array.
    The high density interval are the values which are most credible and cover (1-alpha) of the whole distribution
    Args:
        arr : 1-D array, the data
        alpha: significance. A number between 0 and 1.
    Returns:
        High Density Interval
    """

    return pm.hdi(arr, alpha=alpha)

def max_post_estimate(df_post: pd.DataFrame) -> pd.Series:
    """
    Estimates the value that Maximizes the Posteriori Joint Probability Density Function (MAP)
    Args:
        df_post: A posteriori joint distribution of the model parameters
    Returns:
        MAP
    """

    dp_post = df_post.copy()
    params = dp_post.columns
    kde = stats.gaussian_kde(dp_post.values.T)
    dp_post['pdf'] = kde.pdf(dp_post.values.T)
    map_theta = dp_post[dp_post.pdf == dp_post.pdf.max()].iloc[0][params]
    map_theta.name = 'MAP'

    return map_theta

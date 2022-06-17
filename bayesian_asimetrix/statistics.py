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
import arviz as az

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

    pdfs = prob_density_estimate(df_post)
    map_theta = df_post.iloc[np.where(pdfs == pdfs.max())]
    map_theta['name'] = 'MAP'

    return map_theta

def prob_density_estimate(df_post: pd.DataFrame) -> np.ndarray:
    """
    Estimate the Probability Density Function from the posteriori simulation output
    Args:
        df_post: A posteriori joint distribution of the model parameters
    Returns:
        Probability densities corresponding to each combination of posteriori parameters
    """

    kde = stats.gaussian_kde(df_post.values.T)
    dens = kde.pdf(df_post.values.T)

    return dens

def summary_table(df_post: pd.DataFrame, alpha: float=0.05, round_to: int=2) -> pd.DataFrame:
    """
    Summarizes the information of the posteriori distribution of the model parameters.
    For more information about the ESS and Rhat diagnostics, please refet to:
    https://mc-stan.org/rstan/reference/Rhat.html
    Args:
        df_post (pd.DataFrame):  A posteriori joint distribution of the model parameters
        alpha (float=0.05):  Significance for the High Density Interval. A number between 0 and 1.
        round_to (int=2): Number of decimals used to round results. Defaults to 2. Use None to return the raw numbers
    Returns:
        pd.DataFrame: Summary Table
        - Mode
        - High Density Interval
        - Bulk Effective Sample Size
        - Tail Effective Sample Size
        - Rhat

    """

    # Summmary Table Initialization
    df_summary = pd.DataFrame(index=df_post.columns)

    # Modes
    df_summary['Mode'] = df_post.apply(mode_estimate, raw=True)

    # High Density Intervals
    df_hdi = df_post.apply(lambda x: high_density_interval(x.values, alpha))
    df_hdi.index = [f'HDI_{alpha/2*100}%', f'HDI_{100*(1 - alpha/2)}%']
    df_hdi = df_hdi.transpose()
    df_summary = pd.concat([df_summary, df_hdi], axis=1)

    # Effective Sample Size
    df_summary['Bulk ESS'] = df_post.apply(az.ess, raw=True, **{'method': 'bulk'})
    df_summary['Tail ESS'] = df_post.apply(az.ess, raw=True, **{'method': 'tail'})

    # Rhat
    df_summary['Rhat'] = az.rhat(df_post.values)

    # Rounding
    if round_to:
        df_summary = df_summary.applymap(lambda x: round(x, round_to))

    df_summary
    return df_summary

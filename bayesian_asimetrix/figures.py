"""
Plots for Bayesian Posteriori Diagnostics
Developed by: Asimetrix, Data Science
Author: Ruben D. Vargas
"""

# Python Packages
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bayesian_asimetrix import statistics

def plot_feat_post(df_post: pd.core.frame.DataFrame, feat: str, alpha: float=0.05, rope: list=()):
    """
    Plots the posteriori of a single feature
    The poteriori distribution of a parameter contains all sort of usefull information to diagnose
    and make statistical inference such as hypothesis testing. This is usually complemented with the
    estimation of the Mode and the Hight Density Interval (HDI). In addition, this function allows the user
    to plot the Region of Practial Equivalence (ROPE) to make Hypothesis Testing.
    Args:
        df_post: A posteriori joint distribution of the model parameters
        feat: the name of the parameter that you want to plot
        alpha: the significance of the Hight Density Interval. A number between 0 and 1.
        rope: the Region of Practival Equivalence. If you don't want to plot a ROPE, just ignore this input
        pass an empty tuple.
    Returns:
        Plot the Histogram of the Posteriori Distribution
    """

    # Posteriori Distribution of the parameter
    post_dist = df_post[feat].values

    # Mode Caleculation adn Printting
    print('Mode:', statistics.mode_estimate(post_dist))

    # High Density Interval Estimation
    hdi = statistics.high_density_interval(post_dist, alpha)

    # In case there is a ROPE
    if len(rope) != 0:
      rope_text = [dict(
        showarrow=False, text='ROPE', font_color='gray',
        xref='paper', x=0, yref='paper', y=1.17
      )]
      rope_line = [dict(
          type='line', line=dict(color='gray', width=4),
          xref='x', x0=rope[0], x1=rope[1], yref='y', y0=0, y1=0
      )]
    # In case there isn't a ROPE
    else:
      rope_line = []
      rope_text = []

    # Histogram Plot
    fig = go.Figure(
        data = go.Histogram(x=post_dist),
        layout = go.Layout(
            title=dict(text='A Posteriori Distribution'),
            xaxis=dict(title=feat),
            yaxis=dict(showticklabels=False),
            template='plotly_white',
            height=350,
            bargap=0.1,

            # HDI Plot
            annotations = [dict(
                showarrow=False, text='HDI', font_color='red',
                xref='paper', x=0, yref='paper', y=1.25
            )] + rope_text,
            shapes = [dict(
                type='line', line=dict(color='red', width=5),
                xref='x', x0=hdi[0], x1=hdi[1], yref='y', y0=0, y1=0
            )] + rope_line

        )
    )
    fig.show()
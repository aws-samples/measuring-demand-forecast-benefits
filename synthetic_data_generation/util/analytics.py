# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for analysing synthetic dataset during generation"""
# Python Built-Ins:
from typing import Dict, List, Optional, Union

# External Dependencies:
from matplotlib import pyplot as plt
import pandas as pd


def agg_df_features(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, Union[str, List[str]]]],
    drop_aggregated: bool = True,
) -> None:
    """Aggregate features of a DataFrame (in-place) by string concatenation

    Parameters
    ----------
    df :
        The DataFrame to be modified *IN PLACE*
    config :
        A dictionary of configurations keyed by target (aggregated) field name. Each aggregation
        config is itself a dict containing: `features` (the list of input column names) and
        optionally `sep` (the string separator to be used when concatenating fields).
    drop_aggregated :
        Set `False` to keep the affected original features in the DataFrame. By default (`True`),
        these will be dropped so only the aggregated features (and any columns not affected by
        aggregation) will remain.
    """
    aggregated_features = set()
    for aggname, cfg in config.items():
        df_col_names = cfg["features"]
        if len(df_col_names) > 1:
            lead_col = df[df_col_names[0]]
            aggd_col = lead_col.str.cat(df[df_col_names[1:]], sep=cfg.get("sep"))
            df.loc[:, aggname] = aggd_col
        else:
            df.loc[:, aggname] = df[df_col_names[0]]
        aggregated_features.update(df_col_names)

    if drop_aggregated:
        df.drop(columns=aggregated_features, inplace=True)
    return  # Return None to clarify that modification is in-place.


def log_log_plot(
    value_counts: pd.Series,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xnorm: bool = False,
    **kwargs,
) -> plt.Axes:
    """Generate a log-log frequency analysis

    These log-log plots help you answer questions like "how much of my dataset is at least [Y]".
    For example, "how many of my items have at least N days with sales" or "how many of my items
    sold at least N units". These are useful for characterising *sparsity* in datasets - to
    understand what proportion of items/groups meets the typical data density for a model to work
    well.

    The output is a line chart with negative slope, proceeding from ({minimum value}, {total number
    of items in the dataset}) to ({maximum value}, {number of items that exactly equal that value})

    Parameters
    ----------
    value_counts :
        pandas `value_counts` result for the underlying list you want to characterize.
    xlabel :
        Optional X axis label for the chart
    ylabel :
        Optional Y axis label for the chart
    title :
        Optional title for the chart
    xnorm :
        Set `True` to display X axis as a 0-1 proportion of the dataset. Default `False` displays
        absolute number of items. Useful for comparing datasets of different sizes.
    **kwargs :
        Any additional keyword args are passed through to pyplot `plot()` function.

    Returns
    -------
    ax :
        pyplot Axes for the generated graph.
    """
    # Produce reverse-sorted index of values (e.g. total sales, record counts), with to the
    # cumulative number of items meeting or exceeding each one:
    counts_cumsum = value_counts.sort_index(ascending=False).cumsum()
    if xnorm:
        counts_cumsum /= max(counts_cumsum)
    ax = plt.gca()
    ax.plot(counts_cumsum, counts_cumsum.index, **kwargs)
    ax.set_xscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="both", which="minor")
    return ax

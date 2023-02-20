# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for evaluating and comparing forecasts"""
# Python Built-Ins:
from datetime import datetime
import re

# External Dependencies:
import pandas as pd


def filter_to_period(
    df: pd.DataFrame,
    period_start: datetime,
    period_end: datetime,
    timestamp_col_name: str = "timestamp",
) -> pd.DataFrame:
    """Filter a DataFrame to a particular date/time range (including start, excluding end)

    Handles string, datetime, or period type timestamp columns, and checks that the result is not
    empty.
    """
    n_raw = len(df)
    if timestamp_col_name not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col_name}' not found in DataFrame")

    timestamp_dtype = df[timestamp_col_name].dtype
    if pd.api.types.is_datetime64_any_dtype(timestamp_dtype):
        # DataFrame timestamp column is already parsed as datetimes
        result = df[
            (df[timestamp_col_name] >= period_start) & (df[timestamp_col_name] < period_end)
        ]
    elif pd.api.types.is_period_dtype(timestamp_dtype):
        # DataFrame timestamp column is a 'period' which works a bit differently in Pandas:
        start = pd.Period(period_start, freq=timestamp_dtype.freq)
        end = pd.Period(period_end, freq=timestamp_dtype.freq)
        result = df[(df[timestamp_col_name] >= start) & (df[timestamp_col_name] < end)]
    else:
        # Assume DF timestamp column is strings, but need to check format
        N_TS_FORMAT_SAMPLES = 5
        # TODO: Ignore null values
        ts_tests = df[timestamp_col_name].iloc[:N_TS_FORMAT_SAMPLES]
        ts_tests_lens = tuple(len(txt) for txt in ts_tests)
        if len(set(ts_tests_lens)) != 1:
            raise ValueError(
                "Sampled '%s' values from dataframe have inconsistent length: Cannot infer "
                "date/time format (min %s to max %s)"
                % (
                    timestamp_col_name,
                    min(ts_tests_lens),
                    max(ts_tests_lens),
                )
            )
        if all(map(lambda txt: re.match("\d{4}-\d{2}-\d{2}T", txt), ts_tests)):
            # ISO format e.g. 2000-01-01T12:00:00 (maybe with timezone)
            result = df[
                (df[timestamp_col_name] >= period_start.isoformat())
                & (df[timestamp_col_name] < period_end.isoformat())
            ]
        elif all(map(lambda txt: re.match("\d{4}-\d{2}-\d{2} ", txt), ts_tests)):
            # ISO-like format with space instead of T e.g. 2000-01-01 12:00:00
            # ISO format e.g. 2000-01-01T12:00:00 (maybe with timezone)
            result = df[
                (df[timestamp_col_name] >= period_start.isoformat().replace("T", " "))
                & (df[timestamp_col_name] < period_end.isoformat().replace("T", " "))
            ]
        elif (ts_tests_lens[0] < 11) and all(
            map(lambda txt: re.match("\d{4}(-\d{2}){0,2}", txt), ts_tests)
        ):
            # Truncated ISO-like date e.g. just YYYY-MM or even YYYY.
            start_trunc = period_start.isoformat()[: ts_tests_lens[0]]
            end_trunc = period_end.isoformat()[: ts_tests_lens[0]]
            print(
                "DataFrame has truncated timestamps: Filtering from %s to %s"
                % (start_trunc, end_trunc)
            )
            result = df[
                (df[timestamp_col_name] >= start_trunc) & (df[timestamp_col_name] < end_trunc)
            ]
        else:
            raise ValueError(
                "Couldn't infer date/time format of column '%s' for filtering. Got samples: %s"
                % (timestamp_col_name, ts_tests)
            )

    n_result = len(result)
    if n_result == 0:
        raise ValueError(
            "Filtering dataframe by %s <= [%s] < %s got no results. Check your filter expression"
            % (period_start, timestamp_col_name, period_end)
        )
    print(f"Time period filter kept {n_result} out of {n_raw} records")
    return result.copy()

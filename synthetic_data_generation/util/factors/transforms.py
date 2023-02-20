# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Experimental transformations that can be applied on certain timeseries-generator Factors

WARNING: These transforms will only map to originals on factors that are deterministic at the point
they're transformed! If your factor employs random numbers in generate() and doesn't re/seed the RNG
with the same static value each time, then your original factor and the transformed copy won't
correspond to each other after generation.
"""
# Python Built-Ins:
from copy import deepcopy
from types import MethodType
from typing import Optional, Union

# External Dependencies:
from pandas import DataFrame, Timestamp
from timeseries_generator import BaseFactor


def invert_factor(factor: BaseFactor) -> BaseFactor:
    """Create a deep copy of `factor` which generate()s the inverse of the original"""
    factor = deepcopy(factor)
    generate_original = factor.generate

    def generate(
        self,
        start_date: Optional[Union[Timestamp, str, int, float]],
        end_date: Optional[Union[Timestamp, str, int, float]] = None,
        *args,
        **kwargs,
    ) -> DataFrame:
        df = generate_original(start_date, end_date, *args, **kwargs)
        df[self.col_name] = 1.0 / df[self.col_name]
        return df

    factor.generate = MethodType(generate, factor)
    return factor


def scale_factor(factor: BaseFactor, scale: float = 1.0, base: float = 0.0) -> BaseFactor:
    """Create a deep copy of `factor` which generate()s a scaled version of the original

    `base` is the static offset around which scaling should be applied (e.g. typical choices 0 or 1)
    """
    factor = deepcopy(factor)
    generate_original = factor.generate

    def generate(
        self,
        start_date: Optional[Union[Timestamp, str, int, float]],
        end_date: Optional[Union[Timestamp, str, int, float]] = None,
        *args,
        **kwargs,
    ) -> DataFrame:
        df = generate_original(start_date, end_date, *args, **kwargs)
        if base == 0:
            df[self.col_name] = df[self.col_name] * scale
        else:
            df[self.col_name] = base + ((df[self.col_name] - base) * scale)
        return df

    factor.generate = MethodType(generate, factor)
    return factor

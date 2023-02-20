# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Typed classes for configuring our dataset generation

Defining classes can help to cut user-side boilerplate and reduce errors, versus defining everything
with nested dicts.
"""
# Python Built-Ins:
from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional

# External Dependencies
from numpy import product
from numpy.random import default_rng
from pandas import Timestamp


class SeasonalityConfig:
    """Config for seasonality of a product"""

    AMPLITUDE_DEFAULT: ClassVar[float] = 0.2
    PHASE_ANNUAL_SUMMER: ClassVar[float] = -90
    PHASE_ANNUAL_WINTER: ClassVar[float] = +90
    WAVELENGTH_ANNUAL: ClassVar[float] = 365.25
    WAVELENGTH_2SEASONS: ClassVar[float] = 365.25 / 2

    def __init__(
        self,
        amplitude: float = 0.2,
        mean: float = 1.0,
        phase: float = -90,
        wavelength: float = 365.25,
    ):
        """Create a SeasonalityConfig

        Parameters
        ----------
        amplitude :
            Size of variation due to seasonality. See class var AMPLITUDE_DEFAULT for default value.
        mean :
            Center point of variation. You probably want to leave this as the default 1.0.
        phase :
            Phase offset of seasonality in days. See class vars PHASE_* for common phase values
            appropriate for various wavelengths.
        wavelength :
            Period of seasonality in days. See class vars WAVELENGTH_* for common values such as
            annual or twice-annual seasonality.
        """
        self.amplitude = amplitude
        self.mean = mean
        self.phase = phase
        self.wavelength = wavelength

    def to_sinusoidal_factor_config(self):
        """Convert this config to a dict ready to use in feature_values of SinusoidalFactor"""
        return {
            "amplitude": self.amplitude,
            "mean": self.mean,
            "phase": self.phase,
            "wavelength": self.wavelength,
        }


class BaseCostConfig:
    """Config for base unit cost behaviour of a product"""

    def __init__(
        self,
        initial: Optional[float] = None,
        final_delta: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Create a BaseCostConfig

        Parameters
        ----------
        initial :
            Optional initial unit cost at start of the time period. Randomly generated via default
            exponential distribution if not provided.
        final_delta :
            Optional amount the unit cost should have shifted by the end of the time period
            (positive or negative). Uniformly random between +/- `initial / 2` if not provided.
        rng :
            Optionally provide your own random number generator for defaulting values.
        """
        if (initial is None or final_delta is None) and rng is None:
            rng = default_rng()

        self.initial = rng.exponential() if initial is None else initial
        self.final_delta = (
            rng.uniform(-self.initial / 2, self.initial / 2) if final_delta is None else final_delta
        )

    def to_linear_trend_config(self):
        """Convert this config to a dict ready to use in feature_values of LinearFactor"""
        return {
            "coef": self.final_delta,
            "offset": self.initial,
        }


class ProductConfig:
    """Config for a product"""

    def __init__(
        self,
        base_cost: Optional[BaseCostConfig] = None,
        seasonality: Optional[SeasonalityConfig] = None,
    ):
        """Create a ProductConfig

        Parameters
        ----------
        base_cost :
            The underlying unit cost of the product
        seasonality :
            The demand seasonality of the product
        """
        self.base_cost = base_cost or BaseCostConfig()
        self.seasonality = seasonality or SeasonalityConfig()


def generation_size_diagnostic(
    features: Dict[str, List[str]],
    factors: Dict[str, Any],
    start_date: Timestamp,
    end_date: Timestamp,
) -> None:
    """Print a summary of the overall dataset size/shape to be generated"""
    n_feature_perms = product([len(v) for v in features.values()])
    print("Generating:")
    print(f" - {n_feature_perms} total time-series")
    print(f" - {(end_date - start_date).days / 365:.2f} years of historical data")
    print(
        "Total daily data points: {:.3f} million".format(
            n_feature_perms * (end_date - start_date).days / 1000000
        )
    )
    print("\n".join(["", "FACTORS", "-------", *factors.keys()]))
    print("\n".join(["", "FEATURES", "--------", *features.keys()]))

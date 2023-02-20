# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Python Built-Ins:
from itertools import product
from typing import Any, Dict, List, Optional, Union

# External Dependencies:
import numpy as np
from pandas import DataFrame, Timestamp
from timeseries_generator import BaseFactor
from timeseries_generator.utils import get_cartesian_product


class RandomCompositeFeatureFactor(BaseFactor):
    """A random factor for unique combinations of multiple features

    This class is similar to timeseries_generator.RandomFeatureFactor, but generates random values
    independently for combinations of multiple features, rather than the alternative of layering
    multiple RandomFeatureFactors on different Features.
    """

    def __init__(
        self,
        feature_values: Dict[str, List[Any]],
        min_factor_value: float = 1.0,
        max_factor_value: float = 10.0,
        col_name: str = "random_feature_factor",
    ):
        """Create a RandomCompositeFeatureFactor

        Parameters
        ----------
        feature_values:
            Values (labels) by feature name.
        min_factor_value:
            Minimum factor value.
        max_factor_value:
            Maximum factor value.
        col_name:
            Column name to create for this factor in the generation output.

        Examples
        --------
        Create a factor for every combination of 'store' and 'country' in our list:
            >>> rff = RandomCompositeFeatureFactor(
            ...     feature_values={
            ...         "country": ["country_1", "country_2"],
            ...         "store": ["store_1", "store_2"],
            ...     },
            ...     min_factor_value=1,
            ...     max_factor_value=10
            ... )
        """
        super().__init__(col_name=col_name, features=feature_values)

        self._feature_values = feature_values
        if min_factor_value > max_factor_value:
            raise ValueError(
                f'min_factor_value: "{min_factor_value}" > max_factor_value: "{max_factor_value}"'
            )
        self._min_factor_value = min_factor_value
        self._max_factor_value = max_factor_value

    def generate(
        self,
        start_date: Union[Timestamp, str, int, float],
        end_date: Optional[Union[Timestamp, str, int, float]] = None,
    ) -> DataFrame:
        dr: DataFrame = self.get_datetime_index(start_date=start_date, end_date=end_date).to_frame(
            index=False, name=self._date_col_name
        )

        # calculate product of all provided features and their values:
        factor_df = DataFrame(
            product(*self._feature_values.values()),
            columns=[k for k in self._feature_values.keys()],
        )
        # generate a random factor value for each feature combination:
        # rand_value = min + ((max - min) * value)
        factor_df[self._col_name] = self._min_factor_value + (
            (self._max_factor_value - self._min_factor_value) * np.random.random(len(factor_df))
        )

        # cartesian product of factor df and datetime df
        return get_cartesian_product(dr, factor_df)

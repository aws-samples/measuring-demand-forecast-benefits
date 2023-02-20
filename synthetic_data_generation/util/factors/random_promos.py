# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Python Built-Ins:
from itertools import product
from math import modf
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# External Dependencies:
import numpy as np
from pandas import DataFrame, Timedelta, Timestamp
from timeseries_generator import BaseFactor
from timeseries_generator.utils import get_cartesian_product


class RandomPromotionsFactor(BaseFactor):
    """A *demand side* factor for randomly generated promotion events

    Promotions are randomly generated for each permutation of `feature_values`. For each
    permutation/item, the gaps between promotions are sampled from a Poisson process with
    expectation of `promo_rate` days. The duration of each promotion is sampled from a Poisson
    distribution with expectation `promo_end_rate` days. The impact of each promotion is sampled
    from an exponential distribution with mean `exp_impact`.

    The output factor is 1.0 when each item/permutation is not on promotion, and 1+I for randomly
    sampled impact I, during a promotion period.
    """

    def __init__(
        self,
        feature_values: Dict[str, List[Any]],
        promo_rate: float = 365.25 / 3,
        promo_end_rate: float = 7.0,
        exp_impact: float = 0.3,
        col_name: str = "random_promos_factor",
        random_seed: Optional[Union[int, List[int]]] = None,
    ):
        """Create a RandomPromotionsFactor

        Parameters
        ----------
        feature_values :
            Present values (labels) by feature name, for the features this factor should apply to.
        promo_rate :
            Expectation (Lambda parameter) of the exponential distribution generating the gaps
            between promotions for each item. For example 100 = on average, there are 100 days
            between promotion periods.
        promo_end_rate :
            Expectation (Lambda parameter) of the exponential distribution generating the durations
            of promotions for each item. For example 7 = on average, a promo period lasts 7 days.
        exp_impact :
            Expectation (Lambda parameter) of the exponential distribution generating each
            promotion's positive effect on sales. For example 0.3 = the average promotion will cause
            30% sales uplift.
        col_name :
            Name of the output factor column in the final dataframe
        random_seed :
            Optional explicit seed for random number generation. If not provided, a seed based on
            the current `time.time()` at __init__ invocation will be used.
        """
        super().__init__(col_name=col_name, features=feature_values)

        self._feature_values = feature_values
        self._promo_rate = promo_rate
        self._promo_end_rate = promo_end_rate
        self._exp_impact = exp_impact

        # Generate the random states at the point of __init__, not generate(), so that this factor
        # can be inverted correctly (generate() returns same each time).
        if random_seed is None:
            curr_time_subsec, curr_time_secs = modf(time.time())
            random_seed = [round(curr_time_secs), round(curr_time_subsec * 10**6)]
        self._random_seed = random_seed

    def _load_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._random_seed)

    def gen_random_params(
        self,
        rng: np.random.Generator,
    ) -> Generator[Tuple[int, int, float], None, None]:
        """Iterate random (time-to-start, duration, impact) tuples for promo event generation

        TODO: Batch the actual random number generation here for efficiency
        """
        while True:
            yield (
                round(rng.exponential(self._promo_rate)),
                max(1, round(rng.exponential(self._promo_end_rate))),
                1.0 + rng.exponential(self._exp_impact),
            )

    def generate(
        self,
        start_date: Union[Timestamp, str, int, float],
        end_date: Optional[Union[Timestamp, str, int, float]] = None,
    ) -> DataFrame:
        dt_index = self.get_datetime_index(start_date=start_date, end_date=end_date)
        dt_index_start = min(dt_index)
        dt_index_end = max(dt_index)

        # calculate product of all provided features and their values:
        factor_df = DataFrame(
            product(*self._feature_values.values()),
            columns=[k for k in self._feature_values.keys()],
        )
        factor_df[self._col_name] = 1.0  # Init this factor to 1.0 everywhere
        result = get_cartesian_product(
            dt_index.to_frame(index=False, name=self._date_col_name), factor_df
        ).set_index([self._date_col_name] + [k for k in self._feature_values.keys()])

        # For each feature combination, generate promotions until the target time period is
        # exceeded:
        random_generator = self.gen_random_params(self._load_rng())
        for index, factor_record in factor_df.iterrows():
            cursor = dt_index_start
            events = []
            while cursor <= dt_index_end:
                next_event = next(random_generator)
                start = cursor + Timedelta(days=next_event[0])
                if start > dt_index_end:
                    break  # Can't add the event - it already starts too late.
                end = start + Timedelta(days=next_event[1] + 1)
                result.loc[
                    tuple([slice(start, end)] + [*factor_record][:-1]), self._col_name
                ] = next_event[2]
                events.append(next_event)
                cursor = end

        return result.reset_index()

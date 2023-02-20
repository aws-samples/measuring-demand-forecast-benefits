# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Python Built-Ins:
from typing import Dict, List, Optional, Union

# External Dependencies:
from pandas import DataFrame, Period, PeriodIndex, Timedelta, Timestamp
from timeseries_generator.external_factors import ExternalFactor


class ExternalDateAggregatedFactor(ExternalFactor):
    """timeseries-generator factor based on pre-existing date-aggregated data"""

    def __init__(
        self,
        data_df: DataFrame,
        col_name: str,
        features: Optional[Dict[str, List[str]]] = None,
        date_col_name: str = "date",
        apply_to_all: bool = False,
        min_date: Optional[Union[Timestamp, str, int, float]] = None,
        max_date: Optional[Union[Timestamp, str, int, float]] = None,
    ):
        """Create an ExternalDateAggregatedFactor

        Parameters
        ----------
        data_df :
            A source dataframe that's been aggregated by date/time (has a PeriodIndex, or a
            MultiIndex of which one level is of type Period)
        col_name :
            Column name to generate for this factor in the output
        features :
            Values (labels) by feature name
        date_col_name :
            Name of the date column (or index level) in `data_df`
        apply_to_all :
            As per parent class - TODO: Check this is applied correctly
        min_date :
            Start date beyond which data cannot be generated. If not provided, this will be inferred
            from the `data_df` start date.
        max_date :
            End date beyond which data cannot be generated. If not provided, this will be inferred
            from the `data_df` end date.
        """
        if date_col_name in data_df.columns:
            indexed_by_date = False
        elif date_col_name in data_df.index.names:
            indexed_by_date = True
        else:
            raise ValueError(
                "date_col_name '%s' is not in data_df data columns %s or index columns %s"
                % (date_col_name, data_df.columns, data_df.index.names)
            )

        if min_date is None or max_date is None:
            dates = (
                data_df.index.get_level_values(date_col_name)
                if indexed_by_date
                else data_df[date_col_name]
            )
            if min_date is None:
                min_date = min(dates)
                if isinstance(min_date, Period):
                    min_date = min_date.start_time
            if max_date is None:
                max_date = max(dates)
                if isinstance(max_date, Period):
                    max_date = max_date.end_time

        super().__init__(
            col_name=col_name,
            features=features,
            date_col_name=date_col_name,
            apply_to_all=apply_to_all,
            min_date=min_date,
            max_date=max_date,
        )
        self._indexed_by_date = indexed_by_date
        self._data_df = data_df

    def load_data(self) -> DataFrame:
        """Implement parent abstract method for data loading, but data is already loaded in init"""
        return self._data_df

    def generate(
        self,
        start_date: Union[Timestamp, str, int, float],
        end_date: Optional[Union[Timestamp, str, int, float]] = None,
    ) -> DataFrame:
        if start_date < self.min_date:
            raise ValueError(
                "start_date %s is before this factor's dataset start: %s"
                % (start_date, self.min_date)
            )
        if end_date is None:
            end_date = self.max_date
        elif end_date > self.max_date:
            raise ValueError(
                "end_date %s is after this factor's dataset end: %s" % (end_date, self.max_date)
            )

        data_df = self.load_data()

        # Slice the relevant section of the source data:
        if self._indexed_by_date:
            date_index_levelix = data_df.index.names.index(self._date_col_name)
            date_index = data_df.index.levels[date_index_levelix]
            if isinstance(date_index, PeriodIndex):
                date_slice_start = Period(start_date, freq=date_index.freq)
                date_slice_end = Period(end_date, freq=date_index.freq) + 1
            else:
                date_slice_start = start_date
                date_slice_end = end_date + Timedelta(days=1)
            data_df = data_df.loc[
                tuple(
                    slice(date_slice_start, date_slice_end)
                    if ix == date_index_levelix
                    else slice(None)
                    for ix in range(len(data_df.index.levels))
                ),
                :,  # All columns
            ]
        else:
            raise NotImplementedError(
                "generate() doesn't yet support data_df not *indexed* by date"
            )

        # Expand the aggregated time periods
        if not isinstance(date_index, PeriodIndex):
            # TODO: Should be doable to support others? But not required for our use case
            raise NotImplementedError(
                "generate() doesn't yet support data_df not *indexed* by a pd.PeriodIndex"
            )
        temp_date_col_name = self._date_col_name + "_final"
        datetimes = self.get_datetime_index(start_date=start_date, end_date=end_date)
        datetimes = datetimes.to_series().reset_index(drop=True)
        datetimes = DataFrame(
            {
                temp_date_col_name: datetimes,
                "period": datetimes.dt.to_period("M"),
            },
        ).set_index("period")

        data_df = data_df.reset_index(
            [name for name in data_df.index.names if name != self._date_col_name]
        )
        data_df = data_df.join(datetimes, how="inner")
        data_df = data_df.reset_index(drop=True).rename(
            columns={temp_date_col_name: self._date_col_name}
        )

        return data_df

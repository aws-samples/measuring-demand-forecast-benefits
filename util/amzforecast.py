# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Forecast from Python notebooks"""
# Python Built-Ins:
from datetime import datetime
import json
import re
from typing import Dict, List, Literal, Optional, TypedDict

# External Dependencies:
import boto3  # General-purpose AWS SDK for Python
import pandas as pd  # Tabular data processing tools


forecast = boto3.client("forecast")
s3 = boto3.resource("s3")
s3client = boto3.client("s3")


class AmazonForecastResourceDescDict(TypedDict):
    """Type annotation for Describe* API responses from Amazon Forecast that have a 'Status'

    At the time of writing, the following APIs all meet this spec: DescribeAutoPredictor,
    DescribeDatasetImportJob, DescribeExplainability, DescribeForecast, DescribeMonitor,
    DescribePredictor, DescribeWhatIfAnalysis, DescribeWhatIfForecast, DescribeWhatIfForecastExport

    The following APIs meet the spec but never provide the optional estimated time remaining:
    DescribeExplainabilityExport, DescribeForecastExport, DescribePredictorBacktestExportJob,

    The following APIs mostly meet the spec, but with no ETA and with minor changes to the Status
    value list (remove _STOPPING/_STOPPED statuses, add UPDATE_PENDING, UPDATE_IN_PROGRESS,
    UPDATE_FAILED): DescribeDataset, DescribeDatasetGroup

    Source:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/forecast.html#ForecastService.Client.describe_forecast
    """

    CreationTime: datetime
    EstimatedTimeRemainingInMinutes: Optional[int]  # Only during CREATE_IN_PROGRESS?
    LastModificationTime: datetime
    Status: Literal[
        "ACTIVE",
        "CREATE_PENDING",
        "CREATE_IN_PROGRESS",
        "CREATE_FAILED",
        "CREATE_STOPPING",
        "CREATE_STOPPED",
        "DELETE_PENDING",
        "DELETE_IN_PROGRESS",
        "DELETE_FAILED",
    ]


def is_forecast_resource_ready(desc: AmazonForecastResourceDescDict) -> bool:
    """Check whether the result of an Amazon Forecast Describe* API call is 'Active'/ready

    This function should be usable with the polling progress spinner for pretty much any Amazon
    Forecast Describe* API. Returns True if the resource is ACTIVE, Raises `ValueError` if the
    status includes "FAILED", and returns False otherwise.
    """
    status = desc["Status"]
    if status == "ACTIVE":
        return True
    elif "FAILED" in status:
        raise ValueError(f"Forecast resource failed!\n{desc}")
    return False


def _looks_like_geolocation(value: str) -> bool:
    """Determine whether a field value seems to be in Amazon Forecast GeoLocation format"""
    if re.match("US_\d{5,}", value):
        # US zip code format
        return True
    elif re.match("-?\d+\.\d+_-?\d+\.\d+", value):
        # Lat_Long format
        return True
    return False


def autodiscover_dataframe_schema(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Automatically build an Amazon Forecast schema from a Pandas DataFrame (table)

    Parameters
    ----------
    df :
        Data table to inspect
    overrides :
        Optional mapping of column name to Amazon Forecast 'AttributeType'. If provided, these types
        will be used instead of inferring type for any columns included in the map.
    """
    schema = []
    if not overrides:
        overrides = {}
    for colname in df:
        series = df[colname]
        col_schema = {"AttributeName": colname}
        if colname in overrides:
            col_schema["AttributeType"] = overrides[colname]
        elif pd.api.types.is_datetime64_any_dtype(series) or "timestamp" in colname.lower():
            col_schema["AttributeType"] = "timestamp"
        elif pd.api.types.is_integer_dtype(series):
            col_schema["AttributeType"] = "integer"
        elif pd.api.types.is_numeric_dtype(series):
            col_schema["AttributeType"] = "float"
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            if all(map(_looks_like_geolocation, series.iloc[:10])):
                col_schema["AttributeType"] = "geolocation"
            else:
                col_schema["AttributeType"] = "string"
        else:
            raise ValueError(
                "Couldn't infer Amazon Forecast schema AttributeType for column %s with dtype: %s"
                % (colname, series.dtype)
            )
        schema.append(col_schema)
    return schema


def extract_arn_from_message(msg: str, prefix_regex="arn:aws:forecast:") -> Optional[str]:
    """Try to extract an ARN from a message/string (returning None if no ARN found)"""
    arn = None
    for word in msg.split():
        if re.match(prefix_regex, word):
            arn = word
    return arn


def hash_s3_data(s3uri: str) -> str:
    """Calculate a hash for an object or prefix on S3.

    If s3uri points to an individual object, the S3 ETag is used. If a folder, this function loops
    through all objects under the prefix and returns a string based on the most recent object
    modification timestamp.

    NOTE: This may be slow for large S3 folders, as HeadObject will be called on *every* object
    under the `s3uri` prefix.
    """
    if not s3uri.lower().startswith("s3://"):
        raise ValueError(f"s3uri must start with 's3://'. Got: {s3uri}")
    bucket, _, key = s3uri[len("s3://") :].partition("/")

    try:
        # If the URI is a single object, return its ETag hash:
        headobj = s3client.head_object(
            Bucket=bucket,
            Key=key,
        )
        return json.loads(headobj["ETag"]).replace("-", "_")
    except s3client.exceptions.ClientError as ex:
        # If ResourceNotFound, try to treat it as a prefix. Raise any other errors:
        if ex.response["Error"]["Code"] != "404":
            raise ex

    # Try to look up last modified timestamp for a prefix:
    latest_mod_dt = max(
        objsumm.last_modified for objsumm in s3.Bucket(bucket).objects.filter(Prefix=key)
    )
    return f"mod{datetime.timestamp(latest_mod_dt)}".replace(".", "_")


def create_dataset_import_job_by_hash(**kwargs) -> str:
    """Create a Forecast dataset import jab iff data has changed in Amazon S3

    This function works like boto3 forecast.create_dataset_import_job, but inspects the hash (for
    single files) or last modified timestamp (for folders) of S3 data: Automatically naming the
    import job based on the version of the data and re-using the pre-existing job if the data has
    not changed.

    NOTE: Can be inefficient/slow for large folders - see `hash_s3_data()` for details.
    """
    try:
        data_s3uri = kwargs["DataSource"]["S3Config"]["Path"]
    except KeyError as ke:
        raise ValueError(
            "`DataSource` argument must be a nested dict containing `S3Config.Path`. Got: %s"
            % kwargs.get("DataSource")
        ) from ke

    dataset_name = kwargs["DatasetArn"].partition("/")[2]
    data_hash = hash_s3_data(data_s3uri)
    MAX_JOB_NAME_LEN = 63
    job_name = dataset_name[:50] + "_"  # At most 50 chars from *beginning* of dataset name
    # ...Plus up to remaining allowance from *end* of the hash:
    job_name += data_hash[-(MAX_JOB_NAME_LEN - len(job_name)) :]
    kwargs["DatasetImportJobName"] = job_name

    try:
        resp = forecast.create_dataset_import_job(**kwargs)
        job_arn = resp["DatasetImportJobArn"]
        print(f"Created Dataset Import Job: {job_arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        job_arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if job_arn is None:
            raise ValueError("Couldn't determine ARN of existing Dataset Import Job") from ex
        print(f"Using pre-existing Dataset Import Job: {job_arn}")
    return job_arn


def create_or_reuse_dataset_group(**kwargs) -> str:
    """Thin wrapper over forecast.create_dataset_group(), to re-use existing DSG by name

    Returns the ARN of the new dataset group, or the existing one if one already exists by this
    name.
    """
    try:
        resp = forecast.create_dataset_group(**kwargs)
        arn = resp["DatasetGroupArn"]
        print(f"Created Dataset Group: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing Dataset Group") from ex
        print(f"Using pre-existing Dataset Group: {arn}")
    return arn


def create_or_reuse_dataset(**kwargs) -> str:
    """Thin wrapper over forecast.create_dataset(), to re-use existing dataset by name

    Returns the ARN of the new dataset, or the existing one if one already exists by this name.
    """
    try:
        response = forecast.create_dataset(**kwargs)
        arn = response["DatasetArn"]
        print(f"Created Dataset: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing Dataset") from ex
        print(f"Using pre-existing Dataset {arn}")
    return arn


def create_or_reuse_auto_predictor(**kwargs) -> str:
    """Thin wrapper over forecast.create_auto_predictor(), to re-use existing predictor by name

    Returns the ARN of the new predictor, or the existing one if one already exists by this name.
    """
    try:
        response = forecast.create_auto_predictor(**kwargs)
        arn = response["PredictorArn"]
        print(f"Created AutoPredictor: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing AutoPredictor") from ex
        print(f"Using pre-existing AutoPredictor {arn}")
    return arn


def create_or_reuse_predictor_backtest_export_job(**kwargs) -> str:
    """Thin wrapper over forecast.create_predictor_backtest_export_job(), to re-use existing jobs

    Returns the ARN of the new export job, or the existing one if one already exists by this name.
    """
    try:
        response = forecast.create_predictor_backtest_export_job(**kwargs)
        arn = response["PredictorBacktestExportJobArn"]
        print(f"Created backtest export job: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing backtest export job") from ex
        print(f"Using pre-existing backtest export job {arn}")
    return arn


def create_or_reuse_forecast(**kwargs) -> str:
    """Thin wrapper over forecast.create_forecast(), to re-use existing forecast by name

    Returns the ARN of the new forecast, or the existing one if one already exists by this name.
    """
    try:
        response = forecast.create_forecast(**kwargs)
        arn = response["ForecastArn"]
        print(f"Created forecast: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing forecast") from ex
        print(f"Using pre-existing forecast {arn}")
    return arn


def create_or_reuse_forecast_export_job(**kwargs) -> str:
    """Thin wrapper over forecast.create_forecast_export_job(), to re-use existing export jobs

    Returns the ARN of the new export job, or the existing one if one already exists by this name.
    """
    try:
        response = forecast.create_forecast_export_job(**kwargs)
        arn = response["ForecastExportJobArn"]
        print(f"Created forecast export job: {arn}")
    except forecast.exceptions.ResourceAlreadyExistsException as ex:
        arn = extract_arn_from_message(ex.response["Error"]["Message"])
        if arn is None:
            raise ValueError("Couldn't determine ARN of existing forecast export job") from ex
        print(f"Using pre-existing forecast export job {arn}")
    return arn

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities to set up IAM permissions for Amazon Forecast and Amazon SageMaker

These will only work if the SageMaker notebook role itself has sufficient IAM permissions!
"""
# Python Built-Ins:
import json
from time import sleep

# External Dependencies:
import boto3


def ensure_default_forecast_role(role_name: str = "ForecastRolePOC") -> str:
    """Fetch ARN of a Forecast role allowing full Amazon S3 Access, creating one if needed

    If the role already exists, its permissions are not checked or updated. If a new role is
    created, the 'AmazonS3FullAccess' policy is attached which grants full read and write access to
    all buckets in the AWS Account. WARNING: This is a broad permission set that should not
    typically be used in production environments. Consider scoped-down S3 access by bucket instead.

    Parameters
    ----------
    role_name :
        Name of the role to fetch/create.

    Returns
    -------
    role_arn :
        ARN of the role.
    """
    iam = boto3.client("iam")

    # Try to check if the role exists:
    try:
        role_desc = iam.get_role(RoleName=role_name)
        print(f"Forecast Role '{role_name}' already exists (permissions not checked)")
        return role_desc["Role"]["Arn"]
    except Exception:
        print(f"Creating new role '{role_name}'...")

    # Trust policy should allow Forecast service to assume the role:
    assume_role_policy_doc = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "forecast.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )

    # Create the Role
    create_resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=assume_role_policy_doc,
        Description="Notebook-created execution role for Amazon Forecast",
    )
    role_arn = create_resp["Role"]["Arn"]
    print(
        "Attaching AmazonS3FullAccess policy...\nWARNING: This is a broad permission that you "
        "should consider replacing with narrower permissions for production environments."
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
    )
    print("Waiting for propagation...")
    sleep(15)  # Help ensure any functions called immediately after this have the correct perms
    print("New role ready")
    return role_arn

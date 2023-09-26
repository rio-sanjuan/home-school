# Simple script to read in forecasts and actuals, aggregate by SKU and date, then write back to s3

import sys
import json
import argparse
import logging

import boto3
from datetime import date, datetime, timedelta
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions

import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.window import Window

import numpy as np
import pandas as pd
from typing import List

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

spark = SparkSession.builder.getOrCreate()


def run(input_bucket, forecast_path, actuals_path, output_bucket, output_path):
    logging.info("Load Forecast")
    forecast_df = (
        spark.read.format("parquet")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"s3://{input_bucket}/{forecast_path}")
        .repartition(*(["SKU", "sales_channel", "forecast_date"]))
        .groupby(*(["SKU", "sales_channel", "forecast_date"]))
        .agg(F.sum("forecast_quantity").alias("forecast_quantity"))
    )

    forecast_df.createOrReplaceTempView("forecast")

    logging.info("Load Actuals")
    actuals_df = (
        spark.read.format("parquet")
        .option("header", "true")
        .option("inferSchema": "true")
        .load(f"s3://{input_bucket}/{actuals_path}")
        .withColumnRenamed("actuals_date", "forecast_date")
        .groupby(*(["SKU", "sales_channel", "forecast_date"]))
        .agg(F.sum("actuals_quantity").alias("actuals_quantity"))
    )

    actuals_df.createOrReplaceTempView("actuals")

    sql_agg_statement = f"""
        SELECT
            forecast.forecast_date
            , forecast.SKU
            , sum(forecast.forecast_quantity) as forecast_quantity
            , sum(actuals.actuals_quantity) as actuals_quantity
        FROM 
            forecast
        FULL JOIN
            actuals
        ON
            forecast.SKU = actuals.SKU
            AND forecast.sales_channel = actuals.sales_channel
            AND forecast.forecast_date = actuals.forecast_date
        GROUP BY
            forecast.forecast_date
            , forecast.SKU
    """

    logging.info("Aggregating and writing output")
    agg_df = spark.sql(sql_agg_statement)
    agg_df.write.mode("overwrite").parquet(f"s3://{output_bucket}/{output_path}")

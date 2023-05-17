""" Simple AWS Glue template
- Note that pyspark is preinstalled on glue cluster.
- Note that packages like numpy etc. are pre-installed as well.
- Additional packages need to be configured.
"""
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
import sys
from typing import Union, List
from pyspark.sql import DataFrame


def read_parquet(s3_input_path_or_paths: Union[List[str], str],
                 glueContext: GlueContext) -> DataFrame:
    if isinstance(s3_input_path_or_paths, str):
        s3_input_path_or_paths = [s3_input_path_or_paths]
    gdf = glueContext.create_dynamic_frame.from_options(
        connection_type="s3", connection_options={"paths": s3_input_path_or_paths}, format="parquet")
    return gdf.toDF()


def write_parquet(
    df: DataFrame, output_path: str,
    glueContext, partition_by: List[str] = None
):
    gdf = DynamicFrame.fromDF(df, glueContext, "out")
    connection_options = {"path": output_path}
    if partition_by:
        connection_options["partitionKeys"] = partition_by
    gdf.write(connection_type="s3", connection_options=connection_options, format="parquet")


def main(
    my_job_parameter: str,
    s3_input_path: str,
    s3_output_path: str,
    glueContext
):

    df: DataFrame = read_parquet(s3_input_path, glueContext)
    ...  # Do something with df
    write_parquet(df, s3_output_path, glueContext)


if __name__ == "__main__":

    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "MY_JOB_PARAMETER",
            "S3_INPUT_PATH",
            "S3_OUTPUT_PATH"
        ],
    )

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session

    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    print(f"Hello world. Starting the Job with job name {args['JOB_NAME']}")

    my_job_parameter = args["MY_JOB_PARAMETER"]
    s3_input_path = args["S3_INPUT_PATH"]
    s3_output_path = args["S3_OUTPUT_PATH"]

    main(my_job_parameter, s3_input_path, s3_output_path, glueContext)

    CreateSomeEventToLogandMonitorMyJob = ...  # optional: monitor job

    job.commit()

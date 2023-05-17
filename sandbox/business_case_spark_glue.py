""" Simple Glue Job with the business case
Note that the code is almost identical to sandbox/business_case_spark.py.
The only difference is that we use GlueContext instead of SparkContext.

(!) Make sure that IAM role used for execution has full access to s3.
If not, the error message won't be "Access Denied" but something like "No such file or directory".
"""
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
import sys
from typing import Union, List
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
from datetime import datetime


FEATURES = ['n1', 'n2', 'n3', 'n4_1', 'n4_2', 'n4_3']
TARGET = 'target'


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


def merge_dataframes(df_customer_usage: DataFrame,
                     df_customer_master: DataFrame,
                     df_demographics: DataFrame) -> DataFrame:
    """ Merge dataframes """
    df = df_customer_master.join(df_customer_usage, on="customer_id")
    df = df.join(df_demographics, on="postal_code")
    return df


def one_hot_encode_region(df: DataFrame) -> DataFrame:
    """ One hot encode region """
    df = df.withColumn("n4_1", F.when(F.col("region") == F.lit("City"), 1).otherwise(0))
    df = df.withColumn("n4_2", F.when(F.col("region") == F.lit("Country"), 1).otherwise(0))
    df = df.withColumn("n4_3", F.when(F.col("region") == F.lit("Intermediate Area"), 1).otherwise(0))
    return df


def preprocess_data(df_customer_usage: DataFrame,
                    df_customer_master: DataFrame,
                    df_demographics: DataFrame) -> DataFrame:
    """ Preprocess data """
    df = merge_dataframes(df_customer_usage, df_customer_master, df_demographics)
    df = one_hot_encode_region(df)
    return df


def train_random_forest_classifier(df: DataFrame) -> Transformer:
    """ Train random forest classifier """

    assembler = VectorAssembler(inputCols=FEATURES, outputCol='features')
    train_data = assembler.transform(df).select('features', TARGET)

    clf = RandomForestClassifier(featuresCol='features', labelCol=TARGET, numTrees=100, maxDepth=2, seed=42)
    model = clf.fit(train_data)

    return model


def apply_proba(clf: Transformer, df: DataFrame) -> DataFrame:
    """ Apply probability
    Note that the transformer predicts the probability for each class, resulting in a vector of length 2.
    This vector is of type VectorUDT, which is not supported by Spark SQL.
    Therefore, we need to convert it to an array of doubles.
    """

    assembler = VectorAssembler(inputCols=FEATURES, outputCol='features')
    df = assembler.transform(df)
    df = clf.transform(df)

    df = df.withColumn("probability_vec", vector_to_array(F.col("probability")))
    df = df.withColumn("proba", F.element_at(F.col("probability_vec"), 2))

    return df


def apply_business_logic(df: DataFrame) -> DataFrame:
    """ Apply business logic """
    df = df.withColumn("incentive", F.lit("fallback"))
    df = df.withColumn("incentive", F.when(F.col("proba") >= F.lit(0.1), "v01").otherwise(F.col("incentive")))
    df = df.withColumn("incentive", F.when(F.col("proba") >= F.lit(0.2), "v02").otherwise(F.col("incentive")))
    df = df.withColumn("incentive", F.when(F.col("proba") >= F.lit(0.3), "v03").otherwise(F.col("incentive")))
    df = df.withColumn("incentive", F.when(F.col("proba") >= F.lit(0.4), "v04").otherwise(F.col("incentive")))
    df = df.withColumn("incentive", F.when(F.col("proba") >= F.lit(0.5), "v05").otherwise(F.col("incentive")))
    return df


def main(
    s3_customer_master: str,
    s3_customer_usage: str,
    s3_demographics: str,
    s3_model_path: str,
    s3_output_path: str,
    glueContext
):

    df_customer_master = read_parquet(s3_customer_master, glueContext)
    df_customer_usage = read_parquet(s3_customer_usage, glueContext)
    df_demographics = read_parquet(s3_demographics, glueContext)
    model = RandomForestClassificationModel.load(s3_model_path)

    df = preprocess_data(df_customer_usage, df_customer_master, df_demographics)
    df = apply_proba(model, df)
    df = apply_business_logic(df)
    df = df.select("customer_id", "proba", "incentive")

    write_parquet(df, s3_output_path, glueContext)


if __name__ == "__main__":

    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
        ],
    )

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session

    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    print(f"Hello world. Starting the Job with job name {args['JOB_NAME']}")

    BUCKET_NAME = "pycon-lt"
    s3_customer_master = f"s3://{BUCKET_NAME}/data_more/customer_master_data/"
    s3_customer_usage = f"s3://{BUCKET_NAME}/data_more/customer_usage_data/"
    s3_demographics = f"s3://{BUCKET_NAME}/data/demographics/"
    s3_model_path = f"s3://{BUCKET_NAME}/model/"

    ts = str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    s3_output_path = f"s3://{BUCKET_NAME}/output/run_{ts}/"

    main(s3_customer_master, s3_customer_usage,
         s3_demographics, s3_model_path, s3_output_path,
         glueContext)

    job.commit()

from pyspark.sql import SparkSession


if __name__ == "__main__":

    spark = SparkSession.builder.appName("BusinessCase").getOrCreate()

    df_customer_usage = spark.read.csv("../data/customer_usage_data.csv", header=True, inferSchema=True)
    df_customer_master = spark.read.csv("../data/customer_master_data.csv", header=True, inferSchema=True)
    df_customer_master_wo_target = spark.read.csv("../data/customer_master_data_wo_target.csv", header=True, inferSchema=True)
    df_demographics = spark.read.csv("../data/mocked_demographics.csv", header=True, inferSchema=True)

    df_customer_master.write.parquet("../data/customer_master_data")
    df_customer_usage.write.parquet("../data/customer_usage_data")
    df_customer_master_wo_target.write.parquet("../data/customer_master_data_wo_target")
    df_demographics.write.parquet("../data/demographics")

    spark.stop()

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.ml.base import Transformer
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array


FEATURES = ['n1', 'n2', 'n3', 'n4_1', 'n4_2', 'n4_3']
TARGET = 'target'


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


if __name__ == "__main__":

    spark = SparkSession.builder.appName("BusinessCase").getOrCreate()

    TRAIN = False
    INFERENCE = True

    if TRAIN:
        df_customer_master = spark.read.parquet("../data/customer_master_data")
    else:
        df_customer_master = spark.read.parquet("../data/customer_master_data_wo_target", header=True, inferSchema=True)
    df_customer_usage = spark.read.parquet("../data/customer_usage_data")
    df_demographics = spark.read.parquet("../data/demographics")

    df = preprocess_data(df_customer_usage, df_customer_master, df_demographics)

    if TRAIN:
        """ train the model and save it to disk """
        model = train_random_forest_classifier(df)
        model.save("../data/rf_classifier_model2")

        spark.stop()

    if INFERENCE:
        """ load the model from disk and apply it to the data """
        model = RandomForestClassificationModel.load("../data/rf_classifier_model")
        df = apply_proba(model, df)
        df = apply_business_logic(df)
        df = df.select("customer_id", "proba", "incentive")

    """ just take a look """
    df.printSchema()
    df.show(10)

    spark.stop()

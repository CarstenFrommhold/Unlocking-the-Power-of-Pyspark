from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


FEATURES = ['longitude', 'latitude']
TARGET = 'median_house_value'


def calc_metrics(df: DataFrame, model):
    """Calculate the MSE, MAE, RMSE, and R2 score of the predictions"""
    evaluator = RegressionEvaluator(labelCol='median_house_value', predictionCol='prediction')
    mse = evaluator.evaluate(df, {evaluator.metricName: 'mse'})
    mae = evaluator.evaluate(df, {evaluator.metricName: 'mae'})
    rmse = evaluator.evaluate(df, {evaluator.metricName: 'rmse'})
    r2 = evaluator.evaluate(df, {evaluator.metricName: 'r2'})
    return mse, mae, rmse, r2


def build_model(df: DataFrame):

    # Split the data into train and test sets, with 70% of the data used for training
    df_train, df_test = df.randomSplit([0.7, 0.3], seed=42)

    # Create a vector assembler to combine features into a single vector
    assembler = VectorAssembler(inputCols=FEATURES, outputCol='features')
    train_data = assembler.transform(df_train).select('features', 'median_house_value')
    test_data = assembler.transform(df_test).select('features', 'median_house_value')

    # Fit the linear regression model
    lr = LinearRegression(featuresCol='features', labelCol='median_house_value')
    model = lr.fit(train_data)

    # Use the model to make predictions on the test data
    predictions = model.transform(test_data)

    mse, mae, rmse, r2 = calc_metrics(predictions, model)

    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")


if __name__ == "__main__":

    spark = SparkSession.builder.appName("Housing Regression").getOrCreate()
    df = spark.read.csv("../data/housing_wo_null.csv", header=True, inferSchema=True)
    build_model(df)

    spark.stop()

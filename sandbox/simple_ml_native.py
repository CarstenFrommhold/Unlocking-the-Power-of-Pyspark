import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

FEATURES = ['longitude', 'latitude']
TARGET = 'median_house_value'


def calc_metrics(y_true, y_pred):
    """Calculate the MSE, MAE, RMSE, and R2 score of the predictions"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2


def build_model(df: pd.DataFrame):

    # Split the data into train and test sets, with 70% of the data used for training
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use the model to make predictions on the test data
    y_pred = model.predict(X_test)

    mse, mae, rmse, r2 = calc_metrics(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")


if __name__ == "__main__":

    df = pd.read_csv("../data/housing.csv")
    build_model(df)

import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("../data/housing.csv")
    df.dropna(inplace=True)
    df.to_csv("../data/housing_wo_null.csv", index=False)

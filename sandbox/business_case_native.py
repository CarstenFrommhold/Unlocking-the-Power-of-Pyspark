import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


FEATURES = ['n1', 'n2', 'n3', 'n4_1', 'n4_2', 'n4_3']
TARGET = 'target'


def merge_dataframes(df_customer_usage: pd.DataFrame,
                     df_customer_master: pd.DataFrame,
                     df_demographics: pd.DataFrame) -> pd.DataFrame:
    """ Merge dataframes """
    df = df_customer_master.merge(df_customer_usage, on="customer_id")
    df = df.merge(df_demographics, on="postal_code")
    return df


def one_hot_encode_region(df: pd.DataFrame) -> pd.DataFrame:
    """ One hot encode region """
    df["n4_1"] = (df["region"] == "City").apply(int)
    df["n4_2"] = (df["region"] == "Country").apply(int)
    df["n4_3"] = (df["region"] == "Intermediate Area").apply(int)
    return df


def preprocess_data(df_customer_usage: pd.DataFrame,
                    df_customer_master: pd.DataFrame,
                    df_demographics: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess data """
    df = merge_dataframes(df_customer_usage, df_customer_master, df_demographics)
    df = one_hot_encode_region(df)
    return df


def train_random_forest_classifier(df: pd.DataFrame):
    """ Train random forest classifier """

    X_train = df[FEATURES]
    y_train = df[TARGET]

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    return clf


def apply_proba(clf: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """ Apply probability """
    df["proba"] = clf.predict_proba(df[FEATURES])[:, 1]
    return df


def apply_business_logic(df: pd.DataFrame) -> pd.DataFrame:
    """ Apply business logic """
    df["incentive"] = "fallback"
    df.loc[df["proba"] >= 0.1, "incentive"] = "v01"
    df.loc[df["proba"] >= 0.2, "incentive"] = "v02"
    df.loc[df["proba"] >= 0.3, "incentive"] = "v03"
    df.loc[df["proba"] >= 0.4, "incentive"] = "v04"
    df.loc[df["proba"] >= 0.5, "incentive"] = "v05"
    return df


if __name__ == "__main__":

    TRAIN = False
    INFERENCE = True

    df_customer_usage = pd.read_csv("../data/customer_usage_data.csv")
    df_customer_master = pd.read_csv("../data/customer_master_data.csv")
    df_demographics = pd.read_csv("../data/mocked_demographics.csv")

    df = preprocess_data(df_customer_usage, df_customer_master, df_demographics)

    if TRAIN:
        """ train the model and save it to disk """
        clf = train_random_forest_classifier(df)
        with open("../data/model.pkl", "wb") as f:
            pickle.dump(clf, f)

    if INFERENCE:
        """ load the model from disk and apply it to the data """
        with open("../data/model.pkl", "rb") as f:
            clf = pickle.load(f)
        df = apply_proba(clf, df)
        df = apply_business_logic(df)

    """ just take a look """
    print(df[FEATURES + [TARGET]].corr())
    print(df.head(5))
    print(df["incentive"].value_counts())

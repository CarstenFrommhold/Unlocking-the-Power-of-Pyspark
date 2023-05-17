""" Build synthetic data """
import random
import numpy as np
import pandas as pd

# Define the number of samples
N = 1000


def four_digit_postal_code(number: int) -> str:
    zeros = "0" * (4-len(str(number)))
    return zeros + str(number)


def build_random_row(df_postal_code: pd.DataFrame = None) -> dict:

    n1 = np.random.randn()
    n2 = np.random.randn()
    n3 = np.random.randn()
    postal_code = four_digit_postal_code(random.randrange(1, 9999))
    if df_postal_code is None:
        n4 = random.choice(["City", "Country", "Intermediate Area"])
    else:
        n4 = df_postal_code.loc[df_postal_code.postal_code == postal_code, "region"].values[0]
    n4_1 = 1 if n4 == "City" else 0
    n4_2 = 1 if n4 == "Country" else 0
    n4_3 = 1 if n4 == "Intermediate Area" else 0
    mw = -0.2*n1 + -0.2*n2 + 0.3*n3 + 0.3*n4_1
    coin = mw + 0.5 * np.random.randn()
    target = 1 if coin > 0.6 else 0
    customer_id_ = str(n1) + str(n2) + str(n3) + str(n4) + str(coin)
    customer_id = abs(customer_id_.__hash__())

    return {
        "customer_id": customer_id, "postal_code": postal_code,
        "n1": n1, "n2": n2, "n3": n3, "n4": n4, "n4_1": n4_1, "n4_2": n4_2, "n4_3": n4_3, "target": target
    }


if __name__ == "__main__":

    df_demographic_data = pd.DataFrame([{"postal_code": four_digit_postal_code(n),
                                        "region": random.choice(["City", "Country", "Intermediate Area"])}
    for n in range(1, 9999)])

    df = pd.DataFrame([build_random_row(df_demographic_data) for _ in range(N)])
    print(df[["n1", "n2", "n3", "n4_1", "n4_2", "n4_3", "target"]].corr())
    ...

    WRITE = True
    if WRITE:
        df.to_csv("../data/mocked_customer_data.csv", index=False)
        df_demographic_data.to_csv("../data/mocked_demographics.csv", index=False)
        df[["customer_id", "postal_code", "target"]].to_csv("../data/customer_master_data.csv", index=False)
        df[["customer_id", "postal_code"]].to_csv("../data/customer_master_data_wo_target.csv", index=False)
        df[["customer_id", "n1", "n2", "n3"]].to_csv("../data/customer_usage_data.csv", index=False)

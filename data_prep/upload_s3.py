import boto3
import pandas as pd
from build_synthetic_data import build_random_row
import numpy as np
import os

BUCKET = 'pycon-lt'

if __name__ == '__main__':

    from keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    s3 = boto3.client('s3',
                      region_name='eu-central-1',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    df_demographic_data = pd.read_csv('../data/mocked_demographics.csv',
                                      dtype={'postal_code': str})

    for no in range(1, 500):

        print("Simulate and upload no. ", no)

        N = 10000
        df = pd.DataFrame([build_random_row(df_demographic_data) for _ in range(N)])
        r = str(abs(np.random.randn())).replace('.', '')

        FOLDER = "data_s3"
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)

        df[["customer_id", "postal_code", "target"]].to_parquet(f'{FOLDER}/customer_master_data.parquet', index=False)
        df[["customer_id", "n1", "n2", "n3"]].to_parquet(f'{FOLDER}/customer_usage_data.parquet', index=False)

        with open(f'{FOLDER}/customer_master_data.parquet', 'rb') as f:
            s3.upload_fileobj(f, BUCKET, f'data_more/customer_master_data/customer_master_data_{r}.parquet')

        os.remove(f'{FOLDER}/customer_master_data.parquet')

        with open(f'{FOLDER}/customer_usage_data.parquet', 'rb') as f:
            s3.upload_fileobj(f, BUCKET, f'data_more/customer_usage_data/customer_usage_data_{r}.parquet')

        os.remove(f'{FOLDER}/customer_usage_data.parquet')


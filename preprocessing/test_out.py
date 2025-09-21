import pandas as pd


df = pd.read_parquet("data/train-00000-of-00002-12944970063701d5.parquet")

print(df.head())

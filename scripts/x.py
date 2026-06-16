"""Replace inf / -inf values with NaN in df_proc.parquet (in place)."""

import numpy as np
import pandas as pd

PATH = "/Users/olivier/Developer/learning/credit-scoring/data/processed/df_proc.parquet"

df = pd.read_parquet(PATH)

n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"Found {n_inf} inf/-inf values")

df = df.replace([np.inf, -np.inf], np.nan)

df.to_parquet(PATH, index=False)
print("✅ Saved cleaned parquet to", PATH)

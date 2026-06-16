# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from pathlib import Path

import pandas as pd

from credit_scoring.logger import logger


# LOAD AND SPLIT CORE FUNCTIONS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def load_data(file: Path, num_rows: int | None = None) -> pd.DataFrame:
    """Load a single CSV file and print its shape."""
    df = pd.read_csv(file, nrows=num_rows)
    logger.info(f"🆗 {file.name} loaded (shape = {df.shape[0]:,d} | {df.shape[1]:,d})")
    return df

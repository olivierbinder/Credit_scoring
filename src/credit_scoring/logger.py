# %%  IMPORTS                                                                          .
import logging
import os
import sys

from credit_scoring.config import DIR_ROOT

# %%  LOGGER                                                                           .
logging_str = "%(asctime)s: %(message)s"
log_dir = DIR_ROOT / "logs"
log_filepath = DIR_ROOT / "logs" / "running_logs.log"
os.makedirs(name=log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(filename=log_filepath),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger()

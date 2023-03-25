import logging

import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    df = pd.read_csv("starter/data/census_cleaned.csv")
    logging.info(f"Data Successfully downloaded, shape : {df.shape}")

import logging

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("starter/data/census_cleaned.csv")
    logging.info(f"Data Successfully downloaded, shape : {df.shape}")

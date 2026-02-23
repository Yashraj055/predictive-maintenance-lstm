import pandas as pd

def load_cmapss(file_path):
    """
    Load CMAPSS dataset from txt file
    """
    cols = (
        ["unit", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=cols
    )

    return df

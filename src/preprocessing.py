import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

SCALER_PATH = "models/scaler.pkl"


def preprocess(df, fit_scaler=True):
    """
    Preprocess CMAPSS data.
    - Drops non-sensor columns
    - Fits scaler during training
    - Reuses scaler during evaluation
    """

    df = df.copy()

    # Drop non-sensor columns if present
    drop_cols = []
    for col in ["unit", "cycle"]:
        if col in df.columns:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert to numpy
    data = df.values.astype(np.float32)

    if fit_scaler:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Save scaler
        joblib.dump(scaler, SCALER_PATH)

        return data_scaled

    else:
        # Load fitted scaler
        scaler = joblib.load(SCALER_PATH)
        data_scaled = scaler.transform(data)

        return data_scaled


def create_sequences(data, window_size):
    """
    Convert 2D array into 3D sequences for LSTM
    """
    sequences = []

    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])

    return np.array(sequences, dtype=np.float32)




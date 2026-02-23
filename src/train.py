import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

from src.data_loader import load_cmapss
from src.preprocessing import preprocess, create_sequences


# Config

DATA_PATH = "data/raw/train_FD001.txt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder.h5")

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


# Model definition

def build_lstm_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = LSTM(64, activation="tanh", return_sequences=False)(inputs)

    # Bottleneck
    x = RepeatVector(timesteps)(x)

    # Decoder
    x = LSTM(64, activation="tanh", return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


# Main training pipeline

def main():
    print("🚀 Training started")

    # Load raw data
    print("📂 Loading training data...")
    train_df = load_cmapss(DATA_PATH)
    print("Train shape (raw):", train_df.shape)

    # Preprocess + FIT scaler
    print("🔄 Preprocessing (fit scaler)...")
    train_df = preprocess(train_df, fit_scaler=True)

    # Create sequences
    # Create sequences
    print("🔄 Creating sequences...")
    X_train = create_sequences(train_df, window_size=SEQ_LEN)
    print("X_train shape:", X_train.shape)

  
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    # Build model
    print("🧠 Building model...")
    model = build_lstm_autoencoder(timesteps, n_features)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Train
    print("🔥 Training model...")
    history = model.fit(
        X_train,
        X_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    print("💾 Saving model...")
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    print("🎉 Training complete")


# Entry point

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.data_loader import load_cmapss
from src.preprocessing import preprocess, create_sequences

#  CONFIG 
TEST_PATH = "data/raw/test_FD001.txt"
MODEL_PATH = "models/lstm_autoencoder.h5"
SEQ_LEN = 30
THRESHOLD_PERCENTILE = 95



def main():
    print("🚀 Evaluation started")

    #  Load test data
    print(" Loading test data...")
    test_df = load_cmapss(TEST_PATH)

    #  Preprocess 
    print("🔄 Preprocessing test data...")
    test_df = preprocess(test_df, fit_scaler=False)

    #Create sequences
    print(" Creating sequences...")
    X_test = create_sequences(test_df, window_size=SEQ_LEN)
    print("X_test shape:", X_test.shape)

    # Load trained model 
    print("🧠 Loading trained model...")
    model = load_model(MODEL_PATH, compile=False)

    # Reconstruction
    print("📊 Running reconstruction...")
    X_pred = model.predict(X_test, verbose=0)

    # Reconstruction error
    recon_error = np.mean(np.square(X_test - X_pred), axis=(1, 2))

    # Threshold
    threshold = np.percentile(recon_error, THRESHOLD_PERCENTILE)
    anomalies = recon_error > threshold

    # Console results
    print("📈 Reconstruction error stats:")
    print("Min:", recon_error.min())
    print("Mean:", recon_error.mean())
    print("Max:", recon_error.max())
    print(f"🚨 Threshold ({THRESHOLD_PERCENTILE}th percentile):", threshold)
    print("🚩 Total anomalies detected:", anomalies.sum())


    # PLOTS

    import matplotlib.pyplot as plt

    # Plot 1: Reconstruction error over time
    plt.figure(figsize=(12, 4))
    plt.plot(recon_error, label="Reconstruction Error")
    plt.axhline(threshold, color="red", linestyle="--", label="Anomaly Threshold")
    plt.title("Reconstruction Error over Time")
    plt.xlabel("Time Windows")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reconstruction_error.png")
    plt.close()

    # Plot 2: Error distribution
    plt.figure(figsize=(6, 4))
    plt.hist(recon_error, bins=50)
    plt.axvline(threshold, color="red", linestyle="--")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/error_distribution.png")
    plt.close()

    # Plot 3: Detected anomalies
    plt.figure(figsize=(12, 4))
    plt.plot(recon_error, label="Error")
    plt.scatter(
    np.where(anomalies)[0],
    recon_error[anomalies],
    color="red",
    s=10,
    label="Anomalies"
    )
    plt.legend()
    plt.title("Detected Anomalies")
    plt.tight_layout()
    plt.savefig("results/anomalies.png")
    plt.close()




if __name__ == "__main__":
    main()
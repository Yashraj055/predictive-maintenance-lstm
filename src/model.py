from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense
)
from tensorflow.keras.optimizers import Adam


def build_lstm_autoencoder(timesteps, n_features):
    """
    LSTM Autoencoder for unsupervised fault detection
    """

    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    encoded = LSTM(64, activation="tanh", return_sequences=False)(inputs)

    # Bottleneck
    bottleneck = RepeatVector(timesteps)(encoded)

    # Decoder
    decoded = LSTM(64, activation="tanh", return_sequences=True)(bottleneck)
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return model

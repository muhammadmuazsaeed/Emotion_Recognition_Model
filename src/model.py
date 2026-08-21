from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization


def build_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Build and compile an LSTM-based classifier.

    input_shape : (time_steps, n_mfcc) - shape of one training sample
    num_classes : number of emotion classes to predict
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

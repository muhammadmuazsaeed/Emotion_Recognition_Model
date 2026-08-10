import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import DATASET_DIR, MODELS_DIR, MODEL_PATH, ENCODER_PATH, SCALER_PATH
from src.feature_extraction import build_dataset
from src.model import build_lstm_model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load dataset and extract MFCC features
    # -------------------------------------------------------------
    print("Extracting MFCC features from dataset... this can take a few minutes.")
    X, y = build_dataset(DATASET_DIR)

    if len(X) == 0:
        raise RuntimeError(
            f"No audio files found in '{DATASET_DIR}'. "
            "Download RAVDESS and place the actor folders inside dataset/ before training."
        )

    print(f"Loaded {len(X)} samples, feature shape: {X.shape}")

    # -------------------------------------------------------------
    # 2. Encode labels and scale features
    # -------------------------------------------------------------
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Flatten for scaling, then reshape back to (samples, time_steps, n_mfcc)
    n_samples, n_mfcc, n_frames = X.shape
    X_flat = X.reshape(n_samples, -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(n_samples, n_frames, n_mfcc)  # time_steps, features for LSTM

    # -------------------------------------------------------------
    # 3. Train/test split
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    # -------------------------------------------------------------
    # 4. Build and train the LSTM model
    # -------------------------------------------------------------
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                              num_classes=y_categorical.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------------------------------------------
    # 5. Evaluate: Accuracy, F1-score, MAE
    # -------------------------------------------------------------
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    mae = mean_absolute_error(y_true, y_pred)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"MAE      : {mae:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    # -------------------------------------------------------------
    # 6. Save model, encoder, and scaler for use in the Streamlit app
    # -------------------------------------------------------------
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    main()

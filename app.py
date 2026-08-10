import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder

from src.config import MODEL_PATH, ENCODER_PATH, SCALER_PATH, MAX_PAD_LEN, N_MFCC
from src.feature_extraction import extract_mfcc

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the trained model, label encoder, and scaler once and cache them."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, encoder, scaler


def predict_emotion(file_path, model, encoder, scaler):
    mfcc = extract_mfcc(file_path)                      # shape: (N_MFCC, MAX_PAD_LEN)
    flat = mfcc.reshape(1, -1)
    scaled = scaler.transform(flat)
    reshaped = scaled.reshape(1, MAX_PAD_LEN, N_MFCC)    # (batch, time_steps, features)

    probs = model.predict(reshaped)[0]
    predicted_idx = np.argmax(probs)
    predicted_label = encoder.classes_[predicted_idx]
    return predicted_label, probs, encoder.classes_


def main():
    st.title("🎙️ Speech Emotion Recognition")
    st.write("Upload a speech audio clip (.wav) and the model will predict the emotion being expressed.")

    model, encoder, scaler = load_artifacts()

    if model is None:
        st.warning(
            "No trained model found. Please run `python train.py` first "
            "to train the model on the RAVDESS dataset — this will create "
            "the required files inside the `models/` folder."
        )
        return

    tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎤 Record Live"])

    audio_bytes = None

    with tab_upload:
        uploaded_file = st.file_uploader("Choose a .wav audio file", type=["wav"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getbuffer()
            st.audio(uploaded_file, format="audio/wav")

    with tab_mic:
        st.write("Click the mic, speak a sentence, then click again to stop.")
        recorded_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            sample_rate=48000,
        )
        if recorded_bytes is not None and len(recorded_bytes) > 0:
            audio_bytes = recorded_bytes
            st.audio(recorded_bytes, format="audio/wav")

    if audio_bytes is not None:
        temp_path = os.path.join("dataset", "samples", "_temp_upload.wav")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        with st.spinner("Analyzing audio..."):
            label, probs, classes = predict_emotion(temp_path, model, encoder, scaler)

        st.success(f"Predicted Emotion: **{label.upper()}**")

        # Confidence chart across all emotion classes
        df = pd.DataFrame({"Emotion": classes, "Confidence": probs})
        df = df.sort_values("Confidence", ascending=False)
        st.bar_chart(df.set_index("Emotion"))

        os.remove(temp_path)


if __name__ == "__main__":
    main()
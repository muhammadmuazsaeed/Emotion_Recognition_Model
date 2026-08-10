# 🎙️ Speech Emotion Recognition

Recognize human emotions (happy, sad, angry, neutral, etc.) from speech
audio using deep learning and speech signal processing.

## Overview

This project extracts **MFCC (Mel-Frequency Cepstral Coefficient)**
features from raw speech audio and feeds them into an **LSTM
(Long Short-Term Memory)** neural network, trained from scratch, to
classify the emotion being expressed in the speaker's voice. A
**Streamlit** web app is included for interactively uploading an audio
clip and viewing the predicted emotion.

## Features

- 🎵 MFCC-based audio feature extraction (via `librosa`)
- 🧠 LSTM model trained from scratch — no pretrained weights
- 📊 Evaluation with Accuracy, F1-score, and MAE
- 🖥️ Simple Streamlit UI for real-time predictions
- 📁 Clean, modular project structure

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| Audio processing | Librosa |
| Deep learning | TensorFlow / Keras (LSTM) |
| Evaluation | Scikit-learn |
| Frontend | Streamlit |

## Dataset

This project uses the **[RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)**
(Ryerson Audio-Visual Database of Emotional Speech and Song) dataset —
1,440 speech audio files from 24 actors across 8 emotions: neutral,
calm, happy, sad, angry, fearful, disgust, and surprised.

> The RAVDESS is released under **CC BY-NC-SA 4.0**. If you use it,
> please credit: Livingstone SR, Russo FA (2018) *The Ryerson
> Audio-Visual Database of Emotional Speech and Song (RAVDESS)*.
> PLoS ONE 13(5): e0196391.

## Project Structure

```
emotion-recognition/
├── dataset/
│   ├── samples/          # sample audio files for quick testing
│   └── ...                # place full RAVDESS actor folders here
├── models/                 # trained model, encoder, scaler (generated)
├── src/
│   ├── config.py           # paths, audio params, emotion label map
│   ├── feature_extraction.py  # MFCC extraction logic
│   └── model.py             # LSTM architecture
├── train.py                 # full training pipeline
├── app.py                   # Streamlit frontend
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd emotion-recognition
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download the dataset**
   Download RAVDESS from
   [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
   and place the extracted `Actor_01/`, `Actor_02/`, ... folders inside
   the `dataset/` directory.

3. **Train the model**
   ```bash
   python train.py
   ```
   This extracts MFCC features from every audio file, trains the LSTM
   model, prints Accuracy / F1-score / MAE, and saves the trained
   model to `models/`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the local URL Streamlit prints, upload a `.wav` file, and view
   the predicted emotion.

## How It Works

1. **Preprocessing** — audio is loaded, trimmed of silence, and
   resampled to a fixed duration.
2. **Feature extraction** — 40 MFCC coefficients are extracted per
   clip and padded/truncated to a fixed-length matrix.
3. **Model** — a 2-layer LSTM network learns temporal patterns in the
   MFCC sequence associated with each emotion.
4. **Evaluation** — Accuracy, weighted F1-score, and MAE are reported
   on a held-out test split.
5. **Inference** — the Streamlit app runs the same feature-extraction
   pipeline on uploaded audio and returns the model's prediction with
   a confidence chart.

## Future Improvements

- Add data augmentation (pitch shift, noise injection) to improve
  generalization
- Experiment with CNN + LSTM hybrid architectures
- Add live microphone recording support in the app
- Expand training to include TESS/EMO-DB for more speaker diversity

## License

This project is released under the MIT License. Note that the
RAVDESS dataset itself has its own license (CC BY-NC-SA 4.0) — see
above for attribution requirements.

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # put full RAVDESS here
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_lstm.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Audio / feature extraction parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000        # RAVDESS native sample rate
DURATION = 3.5              # seconds — clips are padded/truncated to this length
N_MFCC = 40                 # number of MFCC coefficients to extract
MAX_PAD_LEN = 174           # fixed number of time frames per MFCC matrix

# ---------------------------------------------------------------------------
# RAVDESS emotion codes -> human-readable labels
# Filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
# e.g. 03-01-06-01-02-01-12.wav -> Emotion code "06" = fearful
# ---------------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTIONS = sorted(set(EMOTION_MAP.values()))

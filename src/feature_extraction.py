import os
import numpy as np
import librosa

from src.config import SAMPLE_RATE, DURATION, N_MFCC, MAX_PAD_LEN, EMOTION_MAP


def extract_mfcc(file_path: str) -> np.ndarray:
    """
    Load a .wav file and extract an MFCC feature matrix of fixed shape
    (N_MFCC, MAX_PAD_LEN). Shorter clips are zero-padded; longer clips
    are truncated, so every sample fed to the model has the same shape.
    """
    # Load audio, resampled/trimmed to a fixed duration
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Trim leading/trailing silence - keeps the model focused on speech
    signal, _ = librosa.effects.trim(signal)

    # Extract MFCCs: shape (N_MFCC, time_frames)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    # Pad or truncate along the time axis so every sample matches MAX_PAD_LEN
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc


def label_from_filename(filename: str) -> str:
    """
    RAVDESS filenames encode the emotion as the 3rd hyphen-separated field:
    e.g. "03-01-06-01-02-01-12.wav" -> emotion code "06" -> "fearful"
    """
    parts = filename.split("-")
    emotion_code = parts[2]
    return EMOTION_MAP[emotion_code]


def build_dataset(dataset_dir: str):
    """
    Walk the dataset directory, extract MFCC features for every .wav file,
    and return (X, y) where X is the feature array and y is the emotion label list.
    """
    X, y = [], []

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            try:
                label = label_from_filename(fname)
            except (IndexError, KeyError):
                # Skip files that don't follow the RAVDESS naming convention
                continue

            fpath = os.path.join(root, fname)
            mfcc = extract_mfcc(fpath)
            X.append(mfcc)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    # Quick manual test: run `python -m src.feature_extraction` from the
    # project root to sanity-check extraction on the sample files.
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "samples")
    for fname in os.listdir(sample_dir):
        if fname.endswith(".wav"):
            path = os.path.join(sample_dir, fname)
            mfcc = extract_mfcc(path)
            label = label_from_filename(fname)
            print(f"{fname} -> label: {label}, MFCC shape: {mfcc.shape}")

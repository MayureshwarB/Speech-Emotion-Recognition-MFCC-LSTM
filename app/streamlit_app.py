"""Streamlit UI for the trained MFCC + LSTM speech-emotion model."""
from pathlib import Path
import io
import json

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "speech_emotion_lstm.keras"
CLASS_METADATA_PATH = ROOT / "models" / "class_names.json"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

DEFAULT_CLASS_METADATA = {
    "class_names": ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"],
    "display_names": {
        "angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
        "neutral": "Neutral", "ps": "Pleasant Surprise", "sad": "Sad"
    },
}


def read_json(path: Path, fallback: dict) -> dict:
    """Read project metadata, falling back to safe defaults if absent or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


CLASS_METADATA = read_json(CLASS_METADATA_PATH, DEFAULT_CLASS_METADATA)
CLASS_NAMES = CLASS_METADATA.get("class_names", DEFAULT_CLASS_METADATA["class_names"])
DISPLAY_NAMES = CLASS_METADATA.get("display_names", DEFAULT_CLASS_METADATA["display_names"])

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a short WAV recording to see the emotion class predicted by an "
    "MFCC + LSTM deep-learning model."
)
st.caption(
    "Research demonstration only. Voice emotion is subjective; predictions are not "
    "psychological or clinical assessments. Model scores are not calibrated confidence."
)


@st.cache_resource
def load_model():
    """Load and cache the trained Keras model."""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def extract_mfcc(audio_bytes: bytes) -> np.ndarray:
    """Match the training pipeline: 40 MFCCs, averaged over time."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), duration=3, offset=0.5)
    if y.size == 0:
        raise ValueError("No audio samples were found in the uploaded file.")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mean_mfcc = np.mean(mfcc.T, axis=0)
    return mean_mfcc.reshape(1, 40, 1).astype(np.float32)


uploaded = st.file_uploader("Choose a WAV audio file", type=["wav"])
if uploaded is not None:
    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    if not MODEL_PATH.exists():
        st.error("The trained model was not found at `models/speech_emotion_lstm.keras`.")
    elif st.button("Predict emotion", type="primary"):
        try:
            model = load_model()
            features = extract_mfcc(audio_bytes)
            scores = np.asarray(model.predict(features, verbose=0)[0], dtype=float)
            if len(scores) != len(CLASS_NAMES):
                st.error(
                    f"The model returned {len(scores)} scores, but the metadata defines "
                    f"{len(CLASS_NAMES)} classes. Check the model and class mapping."
                )
            else:
                best_idx = int(np.argmax(scores))
                best_label = CLASS_NAMES[best_idx]
                best_display = DISPLAY_NAMES.get(best_label, best_label.title())
                st.subheader(f"Predicted emotion: {best_display}")
                st.metric("Top model score (not calibrated confidence)", f"{scores[best_idx] * 100:.2f}%")
                chart_data = {
                    "Model score (%)": {
                        DISPLAY_NAMES.get(label, label.title()): float(score * 100)
                        for label, score in zip(CLASS_NAMES, scores)
                    }
                }
                st.write("Scores across all emotion categories")
                st.bar_chart(chart_data)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the file is a valid WAV and that the installed TensorFlow/Keras "
                "version is compatible with the saved model."
            )

with st.expander("About this model and its evaluation"):
    st.write("**Dataset:** TESS (Toronto Emotional Speech Set), 2,800 audio clips.")
    st.write("**Features:** 40 MFCC coefficients, averaged over the selected audio segment.")
    st.write("**Architecture:** LSTM(128) → Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Softmax(7).")
    metrics = read_json(METRICS_PATH, {})
    if metrics:
        st.metric("Recorded evaluation accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%")
        st.caption(
            "The evaluation partition was also used for validation during training, so this "
            "result should be treated as an in-project result, not a guarantee of real-world performance."
        )

"""
Gradio web interface for the AI Emotion Detection system.
Deploy to HuggingFace Spaces: https://huggingface.co/spaces
Run locally: python app.py
"""

import socket

import gradio as gr
import numpy as np
from PIL import Image
from gradio import utils as gr_utils

# Fix a Gradio backend manager bug in environments where matplotlib loading is blocked.
class _NoOpMatplotlibBackendManager:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

for name in ["MatplotlibBackendMananger", "MatplotlibBackendManager"]:
    if hasattr(gr_utils, name):
        setattr(gr_utils, name, _NoOpMatplotlibBackendManager)

# ── Local fallback emotion classifier ──────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
print("Starting in lightweight local mode.")

EMOTION_INFO = {
    "angry":    {"emoji": "😠", "desc": "Anger detected — frowning expression"},
    "disgust":  {"emoji": "🤢", "desc": "Disgust detected — wrinkled nose"},
    "fear":     {"emoji": "😨", "desc": "Fear detected — wide eyes, raised brows"},
    "happy":    {"emoji": "😊", "desc": "Happiness detected — smiling"},
    "neutral":  {"emoji": "😐", "desc": "Neutral — no strong emotion"},
    "sad":      {"emoji": "😢", "desc": "Sadness detected — downturned mouth"},
    "surprise": {"emoji": "😲", "desc": "Surprise detected — mouth open, wide eyes"},
}

# Lazy model holder (loaded on first prediction)
emotion_pipe = None
_model_loading = False

def ensure_model_loaded():
    """Try to load the HF image-classification pipeline on demand.
    Returns True if model is ready, False otherwise.
    """
    global emotion_pipe, _model_loading
    if emotion_pipe is not None:
        return True
    if _model_loading:
        return False
    _model_loading = True
    try:
        from transformers import pipeline as hf_pipeline
        print("Loading emotion model (lazy)...")
        emotion_pipe = hf_pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=-1,
        )
        print("Model ready!")
        return True
    except Exception as exc:
        print(f"Model load failed (lazy): {exc}")
        emotion_pipe = None
        return False
    finally:
        _model_loading = False


def predict_emotion(upload_image, webcam_image=None):
    """Main prediction function called by Gradio.
    Accepts two inputs: an uploaded image and a webcam capture. If the webcam
    image is provided, it takes precedence over the uploaded image.
    """
    # Prefer webcam capture when available, otherwise use uploaded image
    image = webcam_image if webcam_image is not None else upload_image
    if image is None:
        return "Please upload or capture an image.", {}

    # Convert numpy array (from Gradio) to PIL Image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Ensure RGB (model requires 3 channels)
    pil_image = pil_image.convert("RGB")
    # Attempt to load/run the HF model lazily; fall back quickly if unavailable
    try:
        # Lazy import and loader function defined below; call it here.
        if ensure_model_loaded():
            results = emotion_pipe(pil_image)
            score_map = {r["label"]: r["score"] for r in results}
            scores = {e.capitalize(): float(score_map.get(e, 0.0)) for e in EMOTIONS}

            top = max(results, key=lambda x: x["score"])
            top_label = top["label"]
            top_conf = top["score"] * 100

            info = EMOTION_INFO.get(top_label, {"emoji": "🤔", "desc": ""})
            summary = f"{info['emoji']}  {top_label.upper()} ({top_conf:.1f}%)\n{info['desc']}"
            return summary, scores
    except Exception as exc:
        print(f"Runtime model error: {exc}")

    # Fallback heuristic for immediate local use (improved slightly)
    array = np.array(pil_image)
    gray = np.mean(array, axis=2)
    brightness = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0

    h = array.shape[0]
    upper = gray[: int(h * 0.35), :]
    lower = gray[int(h * 0.55) :, :]
    mouth_var = float(np.std(lower)) / 255.0
    mouth_mean = float(np.mean(lower)) / 255.0
    mouth_delta = mouth_mean - float(np.mean(upper)) / 255.0

    # Debug values can help tune fallback detection when the real model cannot run.
    print(
        f"Fallback stats: brightness={brightness:.2f}, contrast={contrast:.2f}, "
        f"mouth_mean={mouth_mean:.2f}, mouth_var={mouth_var:.2f}, mouth_delta={mouth_delta:.2f}"
    )

    if mouth_mean > 0.55 and mouth_var > 0.03 and mouth_delta > 0.03:
        top_label = "happy"
    elif mouth_var > 0.10 and mouth_delta > 0.04:
        top_label = "surprise"
    elif brightness < 0.38 and mouth_mean < 0.50:
        top_label = "sad"
    elif contrast > 0.28 and mouth_delta < 0.02:
        top_label = "fear"
    else:
        top_label = "neutral"

    scores = {e.capitalize(): 0.0 for e in EMOTIONS}
    scores[top_label.capitalize()] = 1.0
    info = EMOTION_INFO.get(top_label, {"emoji": "🤖", "desc": "Fallback mode"})
    summary = f"{info['emoji']}  {top_label.upper()} (fallback)\n{info['desc']}"
    return summary, scores


# ── Gradio Interface ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_emotion,
    inputs=[
        gr.Image(label="Upload a face photo", type="numpy"),
        gr.Image(label="Or use your webcam (click Capture)", sources=["webcam"], type="numpy"),
    ],
    outputs=[
        gr.Textbox(label="Detected Emotion", lines=2),
        gr.JSON(label="Confidence Scores"),
    ],
    title="AI Facial Emotion Detection",
    description=(
        "Upload a photo of a human face to detect the emotion.\n"
        "Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.\n\n"
        "Model: Vision Transformer (ViT) trained on FER2013 dataset (28,709 real face photos)."
    ),
    examples=[],
)

def find_free_port(start_port=7860, end_port=7899):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free local port available in the range 7860-7899.")


if __name__ == "__main__":
    import os
    requested_port = int(os.environ.get("PORT", 7860))
    port = requested_port if requested_port != 7860 else find_free_port(7860, 7899)
    host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    print(f"Starting local app at http://{host}:{port}")
    demo.launch(
        share=False,
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(),
    )

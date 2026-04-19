import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import time

from transformers import CLIPProcessor, CLIPModel
from deepface import DeepFace


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("Facial Expression Recognition")


# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Using Device: {device}")


# -----------------------------
# Load CLIP Model
# -----------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model.to(device)
    model.eval()
    return model, processor


model, processor = load_clip()


# -----------------------------
# Face Detection
# -----------------------------
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        return face

    return frame


# -----------------------------
# Emotion Prompts
# -----------------------------
emotion_prompt_groups = {
    "Angry": [
        "close-up face angry expression with furrowed brows and tight lips",
        "human face showing anger with intense eyes"
    ],
    "Fear": [
        "scared face wide eyes open mouth fear expression",
        "human face showing fear and anxiety"
    ],
    "Happy": [
        "smiling face with teeth visible happy joyful expression",
        "happy human face with bright eyes"
    ],
    "Sad": [
        "sad face with downcast eyes and frown",
        "human face showing sadness and tears"
    ],
    "Surprise": [
        "surprised face wide eyes raised eyebrows open mouth",
        "shocked human expression"
    ],
    "Disgust": [
        "disgusted face wrinkled nose raised lip",
        "human face showing strong dislike"
    ],
    "Neutral": [
        "neutral human face relaxed expression",
        "calm face no emotion"
    ]
}

emotion_names = list(emotion_prompt_groups.keys())


def get_all_prompts():
    prompts = []
    for emo in emotion_names:
        prompts.extend(emotion_prompt_groups[emo])
    return prompts


def aggregate_scores_max(probs):
    scores = {}
    idx = 0

    for emo in emotion_names:
        k = len(emotion_prompt_groups[emo])
        scores[emo] = float(probs[idx:idx+k].max().item())
        idx += k

    return scores


# -----------------------------
# CLIP Prediction
# -----------------------------
def predict_clip(face_img):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    face_pil = face_pil.resize((224, 224))

    inputs = processor(
        text=get_all_prompts(),
        images=face_pil,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    scores = aggregate_scores_max(probs)
    emo = max(scores, key=scores.get)
    conf = scores[emo] * 100

    return emo, conf, scores


# -----------------------------
# DeepFace Prediction
# -----------------------------
def predict_deepface(face_img):
    try:
        result = DeepFace.analyze(
            face_img,
            actions=['emotion'],
            enforce_detection=False
        )

        emo = result[0]['dominant_emotion']
        conf = result[0]['emotion'][emo]

        return emo.capitalize(), conf

    except:
        return None, 0


# -----------------------------
# Hybrid Logic
# -----------------------------
def hybrid_prediction(face_img):
    clip_emo, clip_conf, clip_scores = predict_clip(face_img)
    df_emo, df_conf = predict_deepface(face_img)

    if df_conf > 60:
        return df_emo, df_conf, "DeepFace", clip_scores
    else:
        return clip_emo, clip_conf, "CLIP", clip_scores


# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Select Mode", ["Upload Image", "Webcam Live"])


# =============================
# IMAGE UPLOAD MODE
# =============================
if mode == "Upload Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)

        face = detect_face(frame)

        emo, conf, model_used, scores = hybrid_prediction(face)

        st.image(face, caption="Detected Face", width=300)
        st.success(f"Emotion: {emo}")
        st.write(f"Confidence: {conf:.2f}%")
        st.write(f"Model Used: {model_used}")

        st.bar_chart({k: v*100 for k, v in scores.items()})


# =============================
# WEBCAM LIVE MODE
# =============================
elif mode == "Webcam Live":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to access webcam")
            break

        face = detect_face(frame)

        emo, conf, model_used, _ = hybrid_prediction(face)

        # Draw result
        cv2.putText(
            frame,
            f"{emo} ({conf:.1f}%) [{model_used}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        FRAME_WINDOW.image(frame, channels="BGR")

        time.sleep(0.1)

    cap.release()


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Major Project | Live Emotion Recognition System (Hybrid AI)")
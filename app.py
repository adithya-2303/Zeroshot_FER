import streamlit as st
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

from models.emotion_cnn import EmotionCNN
from llm.zsl_helper import get_emotion_description, get_embedding


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Zero-Shot Emotion Recognition",
    page_icon="😊"
)

st.title("Zero-Shot Facial Emotion Recognition")
st.write("CNN + LLM Based Emotion Detection")


# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "zsl_model.pth"

ALL_EMOTIONS = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "disgust"   # Unseen emotion
]


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.info(f"Using Device: {device}")


# -----------------------------
# Load CNN Model
# -----------------------------

@st.cache_resource
def load_model():

    model = EmotionCNN(num_classes=6).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Face Detector
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))

])


# -----------------------------
# Prepare LLM Embeddings
# -----------------------------

@st.cache_resource
def load_emotion_embeddings():

    st.write("Loading LLM Knowledge...")

    emb_dict = {}

    for emo in ALL_EMOTIONS:

        desc = get_emotion_description(emo)

        emb = get_embedding(desc)

        emb_dict[emo] = emb

    return emb_dict


emotion_embeddings = load_emotion_embeddings()


# -----------------------------
# Feature Extractor
# -----------------------------

def extract_features(img):

    x = transform(img)

    x = x.unsqueeze(0).to(device)

    with torch.no_grad():

        features = model.forward_embedding(x)

    feats = features.cpu().numpy()[0]
    norm = np.linalg.norm(feats)

    if norm > 0:
        feats = feats / norm

    return feats


# -----------------------------
# Similarity
# -----------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# -----------------------------
# Zero-Shot Prediction
# -----------------------------

def zero_shot_predict(img):

    img_feat = extract_features(img)

    scores = {}

    for emo in ALL_EMOTIONS:

        scores[emo] = cosine_similarity(
            img_feat,
            emotion_embeddings[emo]
        )

    pred = max(scores, key=scores.get)

    return pred, scores


# -----------------------------
# File Upload
# -----------------------------

uploaded = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded:

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, width=300, caption="Uploaded Image")


    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )


    if len(faces) == 0:

        st.error("No face detected. Upload clear face image.")
        st.stop()


    # Crop first face
    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    st.image(face, width=200, caption="Detected Face")


    # Zero-Shot Prediction
    pred, scores = zero_shot_predict(face)


    # Display Result
    st.success(f"Predicted Emotion: {pred.upper()}")


    # Show Scores
    st.subheader("Similarity Scores")

    chart_data = {
        k: float(v)
        for k, v in scores.items()
    }

    st.bar_chart(chart_data)


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.markdown("Major Project | Zero-Shot Emotion Recognition using CNN + LLM")
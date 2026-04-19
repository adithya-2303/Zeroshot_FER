import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

from models.emotion_cnn import EmotionCNN
from llm.zsl_helper import get_emotion_description, get_embedding


# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "zsl_model.pth"

ALL_EMOTIONS = [
    "angry", "fear", "happy",
    "sad", "surprise", "neutral",
    "disgust"   # unseen
]


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Transform
# -----------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -----------------------------
# Load Model
# -----------------------------

model = EmotionCNN(num_classes=6).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


# -----------------------------
# Get Feature Extractor
# -----------------------------

def extract_features(img):

    x = transform(img)

    x = x.unsqueeze(0).to(device)

    with torch.no_grad():

        features = model.conv(x)

        features = features.view(features.size(0), -1)

    return features.cpu().numpy()[0]


# -----------------------------
# Prepare LLM Embeddings
# -----------------------------

print("Loading LLM knowledge...\n")

emotion_embeddings = {}


for emo in ALL_EMOTIONS:

    desc = get_emotion_description(emo)

    emb = get_embedding(desc)

    emotion_embeddings[emo] = emb

    print("Loaded:", emo)


# -----------------------------
# Similarity
# -----------------------------

def cosine(a, b):

    return np.dot(a,b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# -----------------------------
# Zero-Shot Predict
# -----------------------------

def zero_shot_predict(img):

    img_feat = extract_features(img)

    scores = {}


    for emo in ALL_EMOTIONS:

        scores[emo] = cosine(
            img_feat,
            emotion_embeddings[emo]
        )


    return max(scores, key=scores.get), scores


# -----------------------------
# Test Image
# -----------------------------

img = cv2.imread("test.jpg")   # Put disgust image here

pred, scores = zero_shot_predict(img)

print("\nPrediction:", pred)

print("\nAll Scores:")

for k,v in scores.items():

    print(k, ":", round(v,4))
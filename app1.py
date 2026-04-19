import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="Zero-Shot Facial Expression Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("Zero-Shot Facial Expression Recognition")
# st.write("Using CLIP (Vision-Language Model)")


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

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model.to(device)
    model.eval()

    return model, processor


model, processor = load_clip()


# -----------------------------
# Emotion Prompts (LLM Knowledge)
# -----------------------------

emotion_prompt_groups = {
    "Angry": [
        "a face showing anger",
        "an angry expression with furrowed brows",
        "a person with an intense angry face"
    ],
    "Fear": [
        "a face showing fear",
        "a terrified face with wide eyes and open mouth",
        "a frightened expression showing anxiety"
    ],
    "Happy": [
        "a face showing happiness",
        "a joyful smiling face",
        "a happy person beaming with a big smile"
    ],
    "Sad": [
        "a face showing sadness",
        "a sorrowful face with downcast eyes",
        "a person looking sad and tearful"
    ],
    "Surprise": [
        "a face showing surprise",
        "a shocked expression with wide open eyes",
        "a surprised person with raised eyebrows"
    ],
    "Disgust": [
        "a face showing disgust",
        "a repulsed face with wrinkled nose",
        "a person looking disgusted"
    ],
    "Neutral": [
        "a neutral face",
        "a calm expression with no strong emotion",
        "a relaxed neutral facial expression"
    ]
}

emotion_names = list(emotion_prompt_groups.keys())


def get_all_prompt_texts():
    all_prompts = []
    for emotion in emotion_names:
        all_prompts.extend(emotion_prompt_groups[emotion])
    return all_prompts


def aggregate_group_scores(probs):
    grouped_scores = {}
    idx = 0
    for emotion in emotion_names:
        prompts = emotion_prompt_groups[emotion]
        k = len(prompts)
        grouped_scores[emotion] = float(probs[idx:idx+k].mean().item())
        idx += k
    return grouped_scores


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

if uploaded is not None:

    # Load Image
    image = Image.open(uploaded).convert("RGB")

    st.image(image, width=300, caption="Uploaded Image")


    # Prepare Inputs for all prompts and average by emotion
    all_prompts = get_all_prompt_texts()

    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]

    # Aggregate probability per emotion group
    grouped_scores = aggregate_group_scores(probs)

    best_emotion = max(grouped_scores, key=grouped_scores.get)
    emotion = best_emotion
    confidence = grouped_scores[best_emotion] * 100


    # Show Result
    st.success(f"Prediction: {emotion}")
    st.write(f"Confidence: {confidence:.2f}%")


    # -----------------------------
    # Probability Chart
    # -----------------------------

    st.subheader("All Emotion Probabilities")

    chart_data = {
        emo: float(score * 100)
        for emo, score in grouped_scores.items()
    }

    st.bar_chart(chart_data)


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.markdown("Major Project | Zero-Shot Facial Expression Recognition using LLM")
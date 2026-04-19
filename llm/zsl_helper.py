import numpy as np
from sentence_transformers import SentenceTransformer


# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Predefined emotion descriptions (LLM-style)
EMOTION_DESCRIPTIONS = {

    "angry": "Furrowed eyebrows, tight lips, glaring eyes, tense face",

    "fear": "Wide eyes, raised eyebrows, open mouth, tense/terrified expression",

    "happy": "Smiling mouth, raised cheeks, bright eyes, relaxed face",

    "sad": "Drooping eyelids, downturned mouth, teary eyes, slow/low energy expression",

    "surprise": "Raised eyebrows, open mouth, widened eyes",

    "neutral": "Relaxed face, normal eyes, no strong expression",

    "disgust": "Wrinkled nose, raised upper lip, squinted eyes"
}


# ----------------------------------
# Get emotion description
# ----------------------------------

def get_emotion_description(emotion):

    return EMOTION_DESCRIPTIONS[emotion]


# ----------------------------------
# Convert text to embedding
# ----------------------------------

def get_embedding(text):

    emb = embedder.encode(text)
    norm = np.linalg.norm(emb)

    if norm > 0:
        emb = emb / norm

    return emb
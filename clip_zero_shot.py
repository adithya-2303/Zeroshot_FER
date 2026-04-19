import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Load CLIP Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Emotions (Zero-Shot Labels)
emotions = [
    "a face showing anger",
    "a face showing fear",
    "a face showing happiness",
    "a face showing sadness",
    "a face showing surprise",
    "a face showing disgust",
    "a neutral face"
]


# Load Image
image = Image.open("test.jpg")   # Put image here


# Process
inputs = processor(
    text=emotions,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)


# Predict
with torch.no_grad():

    outputs = model(**inputs)

    logits = outputs.logits_per_image

    probs = logits.softmax(dim=1)


# Result
best = torch.argmax(probs).item()

print("Prediction:", emotions[best])
print("Confidence:", probs[0][best].item())
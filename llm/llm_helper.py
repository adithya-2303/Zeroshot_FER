import openai
from sentence_transformers import SentenceTransformer

openai.api_key = "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"

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_emotion_description(emotion):

    prompt = f"""
    Describe the facial features and behavior of a person who is {emotion}.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}]
    )

    text = response['choices'][0]['message']['content']

    return text


def get_embedding(text):

    emb = model.encode(text)
    return emb
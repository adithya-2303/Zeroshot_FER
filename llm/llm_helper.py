import openai
from sentence_transformers import SentenceTransformer

openai.api_key = "YOUR_OPENAI_API_KEY"

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
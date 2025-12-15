import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load variables from .env

# Let the client read OPENAI_API_KEY from environment
client = OpenAI()

def transcribe(file):
    with open(file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return result.text

def ask_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

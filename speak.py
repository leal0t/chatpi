import subprocess
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def speak_audio(text):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    output = "response.wav"
    with open(output, "wb") as f:
        f.write(speech.read())

    # PipeWire/PulseAudio playback
    subprocess.run(["paplay", output])

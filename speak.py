import subprocess
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def speak_audio(text, lcd=None):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="marin",
        input=text
    )

    output = "response.wav"
    with open(output, "wb") as f:
        f.write(speech.read())

    # PipeWire/PulseAudio playback
    proc = subprocess.Popen(["paplay", output])

    if lcd:
        # Animate lips while audio plays
        while proc.poll() is None:
            lcd.show_face(mouth_open=True)
            time.sleep(0.14)
            lcd.show_face(mouth_open=False)
            time.sleep(0.09)
        lcd.show_face(mouth_open=False)
    else:
        proc.wait()

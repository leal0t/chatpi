import subprocess
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)

def speak_audio(text, lcd=None):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="marin",
        input=text
    )

    output = "response.wav"
    with open(output, "wb") as f:
        f.write(speech.read())

    # PipeWire/PulseAudio playback — locked to USB speaker
    proc     = subprocess.Popen(["paplay", "--device=alsa_output.usb-Generic_USB2.0_Device_20121120222012-00.analog-stereo", output])
    deadline = time.time() + 60.0  # never hang more than 60s

    if lcd:
        while proc.poll() is None and time.time() < deadline:
            lcd.show_face(mouth_open=True)
            time.sleep(0.14)
            lcd.show_face(mouth_open=False)
            time.sleep(0.09)
        if proc.poll() is None:
            proc.kill()
        lcd.show_face(mouth_open=False)
    else:
        try:
            proc.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            proc.kill()

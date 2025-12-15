import sounddevice as sd
import wavio

def record_audio(filename="input.wav"):
    samplerate = 16000
    duration = 6  # seconds

    print("Listening... speak now.")

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    wavio.write(filename, audio, samplerate, sampwidth=2)
    return filename

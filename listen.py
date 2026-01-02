import sounddevice as sd
import wavio
import numpy as np

def record_audio(duration=6, filename="input.wav"):
    """
    Record audio from the default microphone for `duration` seconds
    and save it to `filename`. Returns (filename, rms_level).
    """
    samplerate = 16000  # Hz

    print(f"Listening... speak now. ({duration} seconds)")

    # Record mono audio as float32 in [-1.0, 1.0]
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    # Compute RMS loudness (root-mean-square of the waveform)
    # Flatten to 1D just in case
    mono = audio.flatten()
    rms = float(np.sqrt(np.mean(np.square(mono))))

    # Save as 16-bit WAV
    wavio.write(filename, audio, samplerate, sampwidth=2)

    return filename, rms

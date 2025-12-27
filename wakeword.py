import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models

class WakeWordDetector:
    def __init__(self, threshold: float = 0.5):
        download_models()
        self.model = Model(wakeword_models=["hey rhasspy"])
        self.threshold = threshold
        self.samplerate = 16000
        self.frame_duration = 0.5
        self.frame_samples = int(self.samplerate * self.frame_duration)

    def wait_for_wake_word(self):
        print("🎧 Listening for wake word: say 'Hey Rhasspy' to wake Hali...")
        with sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
        ) as stream:
            while True:
                audio, overflowed = stream.read(self.frame_samples)
                if overflowed:
                    print("⚠️ Audio buffer overflowed, continuing...")
                mono = audio[:, 0]
                pcm16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767)
                scores = self.model.predict(pcm16)
                score = scores.get("hey rhasspy", 0.0)
                if score >= self.threshold:
                    print(f"✅ Wake word detected! (score={score:.2f})")
                    return

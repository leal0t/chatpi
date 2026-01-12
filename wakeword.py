import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models


class WakeWordDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        samplerate: int = 16000,
        frame_duration: float = 0.5,
        blocksize: int = 2048,
        latency: str = "low",
        device: int | None = None,
        cooldown_seconds: float = 1.0,
        verbose_overflows: bool = True,
    ):
        """
        Pi Zero 2 W friendly wakeword listener.

        threshold: wakeword score threshold.
        samplerate: openWakeWord expects 16 kHz input.
        frame_duration: seconds per inference frame (0.5 is a good balance).
        blocksize: smaller blocks reduce buffer overflow on slow devices.
        latency: "low" helps keep the input buffer from piling up.
        device: optional sounddevice input device index (None = default).
        cooldown_seconds: minimum time after a detection before listening again.
        verbose_overflows: print overflow warnings (can disable if annoying).
        """
        download_models()

        # If you ever want to force TFLite-only:
        # self.model = Model(wakeword_models=["hey rhasspy"], inference_framework="tflite")
        self.model = Model(wakeword_models=["hey rhasspy"])

        self.threshold = threshold
        self.samplerate = samplerate
        self.frame_duration = frame_duration
        self.frame_samples = int(self.samplerate * self.frame_duration)

        self.blocksize = blocksize
        self.latency = latency
        self.device = device
        self.cooldown_seconds = cooldown_seconds
        self.verbose_overflows = verbose_overflows

        self._last_detect_time = 0.0

    def wait_for_wake_word(self):
        print("🎧 Listening for wake word: say 'Hey Rhasspy' to wake Hali...")

        # float32 stream -> convert to int16 PCM for openWakeWord
        with sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.latency,
            device=self.device,
        ) as stream:
            while True:
                # Read one inference frame
                audio, overflowed = stream.read(self.frame_samples)

                if overflowed and self.verbose_overflows:
                    print("⚠️ Audio buffer overflowed, continuing...")

                # Cooldown: prevents rapid retriggers
                now = time.time()
                if now - self._last_detect_time < self.cooldown_seconds:
                    # yield CPU a bit
                    time.sleep(0.01)
                    continue

                # audio shape is (N, 1)
                mono = audio[:, 0]

                # Convert float32 [-1..1] -> int16 PCM
                pcm16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767)

                # Predict wakeword score
                scores = self.model.predict(pcm16)

                # Some versions return dict-like; be defensive:
                score = 0.0
                if isinstance(scores, dict):
                    score = float(scores.get("hey rhasspy", 0.0))
                else:
                    # If scores is something else, try best-effort:
                    try:
                        score = float(scores["hey rhasspy"])
                    except Exception:
                        score = 0.0

                if score >= self.threshold:
                    self._last_detect_time = now
                    print(f"✅ Wake word detected! (score={score:.2f})")
                    return

                # tiny sleep keeps CPU from pegging at 100%
                time.sleep(0.001)


if __name__ == "__main__":
    # Optional: print devices to find the correct mic index
    # print(sd.query_devices())

    detector = WakeWordDetector(threshold=0.5)
    detector.wait_for_wake_word()

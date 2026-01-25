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

        # ✅ NEW anti-false-wake knobs
        warmup_seconds: float = 2.0,          # ignore audio right after starting to listen
        consecutive_hits_required: int = 2,   # require N frames above threshold
    ):
        download_models()
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

        self.warmup_seconds = warmup_seconds
        self.consecutive_hits_required = max(1, int(consecutive_hits_required))

        self._last_detect_time = 0.0

    def wait_for_wake_word(self):
        print("🎧 Listening for wake word: say 'Hey Rhasspy' to wake Hali...")

        consecutive_hits = 0

        with sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.latency,
            device=self.device,
        ) as stream:

            # ✅ NEW: Warmup discard to avoid immediate false trigger
            # This dumps buffered audio / echo right after we enter wakeword mode.
            warmup_end = time.time() + self.warmup_seconds
            while time.time() < warmup_end:
                # read smaller chunks so we don't block too long
                _chunk, _overflowed = stream.read(self.blocksize)
                time.sleep(0.001)

            while True:
                audio, overflowed = stream.read(self.frame_samples)

                if overflowed and self.verbose_overflows:
                    print("⚠️ Audio buffer overflowed, continuing...")

                # Cooldown: prevents rapid retriggers
                now = time.time()
                if now - self._last_detect_time < self.cooldown_seconds:
                    time.sleep(0.01)
                    continue

                mono = audio[:, 0]
                pcm16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767)

                scores = self.model.predict(pcm16)
                if isinstance(scores, dict):
                    score = float(scores.get("hey rhasspy", 0.0))
                else:
                    try:
                        score = float(scores["hey rhasspy"])
                    except Exception:
                        score = 0.0

                # ✅ NEW: require consecutive hits to avoid single-frame spikes/echo
                if score >= self.threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= self.consecutive_hits_required:
                    self._last_detect_time = now
                    print(f"✅ Wake word detected! (score={score:.2f})")
                    return

                time.sleep(0.001)


if __name__ == "__main__":
    detector = WakeWordDetector(threshold=0.5)
    detector.wait_for_wake_word()

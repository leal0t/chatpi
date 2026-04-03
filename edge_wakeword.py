import time
from collections import deque

import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite


class EdgeWakeWordDetector:
    def __init__(
        self,
        model_path: str,
        samplerate: int = 16000,
        frame_duration: float = 1.0,
        hop_duration: float = 0.20,
        energy_threshold: float = 0.010,
        wakeword_class: int = 0,        # 0=hey_hali, 1=noise, 2=unknown (alphabetical)
        debounce_frames: int = 2,
        cooldown_seconds: float = 2.5,
        smoothing_frames: int = 3,
        confidence_threshold: float = 0.55,  # wake score must exceed this (0.0-1.0)
        confidence_margin: float = 0.10,     # wake must beat best_other by this much
        device: int | None = None,
    ):
        self.samplerate = samplerate
        self.frame_duration = frame_duration
        self.hop_duration = hop_duration
        self.frame_samples = int(frame_duration * samplerate)
        self.hop_samples = int(hop_duration * samplerate)

        self.energy_threshold = energy_threshold
        self.wakeword_class = wakeword_class
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.smoothing_frames = smoothing_frames
        self.confidence_threshold = confidence_threshold
        self.confidence_margin = confidence_margin
        self.device = device

        self._last_detect_time = 0.0
        self._trigger_count = 0
        self._wake_history = deque(maxlen=smoothing_frames)
        self._prediction_history = deque(maxlen=smoothing_frames)
        self._consecutive_wake_frames = 0
        self._frames_seen = 0
        self._warmup_frames = 4
        self._buffer = np.zeros(self.frame_samples, dtype=np.float32)

        # Load model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index  = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_shape  = self.input_details[0]["shape"]
        self.input_dtype  = self.input_details[0]["dtype"]
        self.input_len    = int(self.input_shape[1])

        # ── Input quantization params ────────────────────────────
        # Correct formula: quantized = (float_value / scale) + zero_point
        in_quant = self.input_details[0].get("quantization", (1.0, 0))
        self._in_scale      = in_quant[0] if in_quant[0] != 0 else 1.0
        self._in_zero_point = in_quant[1]

        # ── Output quantization params ───────────────────────────
        # Correct formula: float_value = scale * (quantized - zero_point)
        out_quant = self.output_details[0].get("quantization", (1.0, 0))
        self._out_scale      = out_quant[0] if out_quant[0] != 0 else 1.0
        self._out_zero_point = out_quant[1]

        print("🧠 Edge Impulse wakeword model loaded")
        print(f"   Input  shape : {self.input_shape}")
        print(f"   Input  dtype : {self.input_dtype}")
        print(f"   In  scale    : {self._in_scale}  zero_pt: {self._in_zero_point}")
        print(f"   Output dtype : {self.output_details[0]['dtype']}")
        print(f"   Out scale    : {self._out_scale}  zero_pt: {self._out_zero_point}")
        print(f"   Wakeword cls : {self.wakeword_class} (0=hey_hali,1=noise,2=unknown)")
        print(f"   Conf thresh  : {self.confidence_threshold}")
        print(f"   Conf margin  : {self.confidence_margin}")
        print(f"   Frame samples: {self.frame_samples}")
        print(f"   Hop   samples: {self.hop_samples}")

    # ── Audio prep ────────────────────────────────────────────────
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio → quantized int8 using model's own scale/zero_point."""
        if len(audio) < self.input_len:
            audio = np.pad(audio, (0, self.input_len - len(audio)))
        else:
            audio = audio[:self.input_len]

        if self.input_dtype == np.int8:
            # Correct quantization: q = round(x / scale) + zero_point
            quantized = np.round(audio / self._in_scale) + self._in_zero_point
            audio = np.clip(quantized, -128, 127).astype(np.int8)
        elif self.input_dtype == np.int16:
            audio = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        else:
            audio = audio.astype(self.input_dtype)

        return audio.reshape(1, -1)

    # ── Inference + dequantize ────────────────────────────────────
    def _infer(self, audio_window: np.ndarray) -> np.ndarray:
        """Run inference and return dequantized float probabilities (0.0–1.0)."""
        features = self._prepare_audio(audio_window)
        self.interpreter.set_tensor(self.input_index, features)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_index)[0]
        
        print(f"RAW int8 output: {raw}")   # add this line
        # Dequantize output: real_value = scale * (quantized - zero_point)
        dequantized = self._out_scale * (raw.astype(np.float32) - self._out_zero_point)

        # Softmax → probabilities summing to 1.0
        e = np.exp(dequantized - np.max(dequantized))
        return e / e.sum()

    def _reset_detection_state(self):
        self._trigger_count = 0
        self._wake_history.clear()
        self._prediction_history.clear()
        self._consecutive_wake_frames = 0
        self._frames_seen = 0

    # ── Main loop ─────────────────────────────────────────────────
    def wait_for_wake_word(self):
        print("🎧 Listening for wake word...")

        with sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            device=self.device,
            blocksize=self.hop_samples,
        ) as stream:
            while True:
                chunk, _ = stream.read(self.hop_samples)
                chunk = chunk[:, 0]

                # Slide buffer forward
                self._buffer[:-self.hop_samples] = self._buffer[self.hop_samples:]
                self._buffer[-self.hop_samples:] = chunk

                # Energy gate — skip silence
                rms = np.sqrt(np.mean(self._buffer ** 2))
                #if rms < self.energy_threshold:
                #    self._trigger_count = 0
                #    self._consecutive_wake_frames = 0
                #    continue

                # Cooldown gate
                now = time.time()
                if now - self._last_detect_time < self.cooldown_seconds:
                    continue

                # Inference → probabilities
                probs = self._infer(self._buffer)

                wake_score    = float(probs[self.wakeword_class])
                other_probs   = np.delete(probs, self.wakeword_class)
                best_other    = float(np.max(other_probs))
                predicted_cls = int(np.argmax(probs))
                margin        = wake_score - best_other

                self._frames_seen += 1
                self._wake_history.append(wake_score)
                self._prediction_history.append(predicted_cls)

                smoothed_wake = float(np.mean(self._wake_history))
                wake_votes    = sum(1 for p in self._prediction_history if p == self.wakeword_class)

                is_strong_wake = (
                    predicted_cls == self.wakeword_class
                    and wake_score >= self.confidence_threshold
                    and margin >= self.confidence_margin
                )

                if is_strong_wake:
                    self._consecutive_wake_frames += 1
                else:
                    self._consecutive_wake_frames = 0

                print(f"Probs      : {np.round(probs, 3)}  (hey_hali | noise | unknown)")
                print(f"Wake score : {wake_score:.3f}  best_other: {best_other:.3f}  margin: {margin:.3f}")
                print(f"Predicted  : {predicted_cls}  smoothed: {smoothed_wake:.3f}  votes: {wake_votes}")
                print(f"Streak     : {self._consecutive_wake_frames}  RMS: {rms:.4f}")
                print("----")

                enough_history = len(self._prediction_history) >= 3
                majority_wake  = wake_votes >= 2
                strong_streak  = self._consecutive_wake_frames >= 2
                past_warmup    = self._frames_seen >= self._warmup_frames

                if (
                    past_warmup
                    and enough_history
                    and majority_wake
                    and strong_streak
                    and is_strong_wake
                ):
                    self._trigger_count += 1
                    print(f"🔔 Trigger count: {self._trigger_count}/{self.debounce_frames}")
                else:
                    self._trigger_count = 0

                if self._trigger_count >= self.debounce_frames:
                    print("✅ Wake word detected!")
                    self._last_detect_time = time.time()
                    self._reset_detection_state()
                    return

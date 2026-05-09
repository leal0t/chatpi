import gc
import time

import numpy as np
import sounddevice as sd
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    from ai_edge_litert import interpreter as tflite


# ============================================================
# MFE (Mel Filterbank Energy) preprocessing
# Parameters match Edge Impulse Transfer Learning (Keyword Spotting):
#   Frame length : 0.02s
#   Frame stride : 0.01s
#   Filter number: 40
#   FFT length   : 256
#   Low frequency: 0 Hz
#   Noise floor  : -52 dB
# ============================================================

def compute_mfe(
    audio: np.ndarray,
    samplerate: int = 16000,
    frame_length: float = 0.02,
    frame_stride: float = 0.01,
    num_filters: int = 40,
    fft_length: int = 256,
    low_frequency: float = 0.0,
    noise_floor_db: float = -52.0,
) -> np.ndarray:
    frame_len    = int(round(frame_length * samplerate))
    frame_stride = int(round(frame_stride * samplerate))

    # Frame the signal
    num_frames = 1 + (len(audio) - frame_len) // frame_stride
    indices    = (
        np.tile(np.arange(frame_len), (num_frames, 1))
        + np.tile(np.arange(num_frames) * frame_stride, (frame_len, 1)).T
    )
    frames = audio[indices].astype(np.float32)

    # Hamming window
    frames *= np.hamming(frame_len)

    # FFT → power spectrum
    mag   = np.abs(np.fft.rfft(frames, n=fft_length))
    power = (mag ** 2) / fft_length

    # Mel filterbank
    high_frequency = samplerate / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low  = hz_to_mel(low_frequency)
    mel_high = hz_to_mel(high_frequency)
    mel_pts  = np.linspace(mel_low, mel_high, num_filters + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((fft_length + 1) * hz_pts / samplerate).astype(int)

    fbank = np.zeros((num_filters, fft_length // 2 + 1), dtype=np.float32)
    for m in range(1, num_filters + 1):
        f_left   = bin_pts[m - 1]
        f_center = bin_pts[m]
        f_right  = bin_pts[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    # Apply filterbank → energy
    energy = np.dot(power, fbank.T)

    # Noise floor clipping
    noise_floor_linear = 10.0 ** (noise_floor_db / 10.0)
    energy = np.maximum(energy, noise_floor_linear)

    # Log energy
    energy = np.log(energy)

    # Fixed normalization — preserves energy differences between speech and silence
    energy = np.clip((energy + 12.0) / 12.0, 0.0, 1.0)

    # Free intermediate arrays before returning
    del frames, mag, power, fbank, indices

    return energy.flatten().astype(np.float32)


class EdgeWakeWordDetector:
    def __init__(
        self,
        model_path: str,
        samplerate: int = 16000,
        frame_duration: float = 1.0,        # 1000ms window
        hop_duration: float = 0.20,         # 200ms stride
        energy_threshold: float = 0.010,    # skip silent frames
        wakeword_class: int = 0,            # 0=hey_hali, 1=noise, 2=unknown
        cooldown_seconds: float = 2.0,      # pause after detection
        confidence_threshold: float = 0.40, # minimum hey_hali score
        confidence_margin: float = 0.10,    # must beat next class by this
        max_misses: int = 6,                # clear buffer after this many non-detections
        rest_seconds: float = 30.0,         # pause after buffer clear before resuming
        device: int | None = None,
    ):
        self.samplerate           = samplerate
        self.frame_duration       = frame_duration
        self.hop_duration         = hop_duration
        self.frame_samples        = int(frame_duration * samplerate)
        self.hop_samples          = int(hop_duration * samplerate)
        self.energy_threshold     = energy_threshold
        self.wakeword_class       = wakeword_class
        self.cooldown_seconds     = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.confidence_margin    = confidence_margin
        self.max_misses           = max_misses
        self.rest_seconds         = rest_seconds
        self.device               = device

        self._last_detect_time = 0.0
        self._buffer           = np.zeros(self.frame_samples, dtype=np.float32)

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

        # Output quantization
        out_quant            = self.output_details[0].get("quantization", (1.0, 0))
        self._out_scale      = out_quant[0] if out_quant[0] != 0 else 1.0
        self._out_zero_point = out_quant[1]

        # Input quantization
        in_quant            = self.input_details[0].get("quantization", (1.0, 0))
        self._in_scale      = in_quant[0] if in_quant[0] != 0 else 1.0
        self._in_zero_point = in_quant[1]

        print("🧠 Edge Impulse Transfer Learning wakeword model loaded")
        print(f"   Input  shape      : {self.input_shape}")
        print(f"   Input  dtype      : {self.input_dtype}")
        print(f"   In  scale/zero_pt : {self._in_scale} / {self._in_zero_point}")
        print(f"   Out scale/zero_pt : {self._out_scale} / {self._out_zero_point}")
        print(f"   Wakeword class    : {self.wakeword_class} (0=hey_hali,1=noise,2=unknown)")
        print(f"   Conf threshold    : {self.confidence_threshold}")
        print(f"   Conf margin       : {self.confidence_margin}")
        print(f"   Frame samples     : {self.frame_samples}  ({frame_duration*1000:.0f}ms)")
        print(f"   Hop   samples     : {self.hop_samples}  ({hop_duration*1000:.0f}ms)")
        print(f"   MFE input len     : {self.input_len} features")
        print(f"   Max misses        : {self.max_misses}")

    def _prepare_features(self, audio: np.ndarray) -> np.ndarray:
        features = compute_mfe(audio, samplerate=self.samplerate)

        if len(features) < self.input_len:
            features = np.pad(features, (0, self.input_len - len(features)))
        else:
            features = features[:self.input_len]

        if self.input_dtype == np.int8:
            quantized = np.round(features / self._in_scale) + self._in_zero_point
            features  = np.clip(quantized, -128, 127).astype(np.int8)
        elif self.input_dtype == np.int16:
            features = (features * 32767.0).clip(-32768, 32767).astype(np.int16)
        else:
            features = features.astype(self.input_dtype)

        return features.reshape(1, -1)

    def _infer(self, audio_window: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities [0.0–1.0] for each class."""
        features = self._prepare_features(audio_window)
        self.interpreter.set_tensor(self.input_index, features)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_index)[0]

        dequantized = self._out_scale * (raw.astype(np.float32) - self._out_zero_point)
        e = np.exp(dequantized - np.max(dequantized))
        probs = e / e.sum()

        # Free inference temporaries immediately
        del features, raw, dequantized, e

        return probs

    def wait_for_wake_word(self):
        print("🎧 Listening for wake word...")

        while True:
            try:
                with sd.InputStream(
                    channels=1,
                    samplerate=self.samplerate,
                    dtype="float32",
                    device=self.device,
                    blocksize=self.hop_samples,
                ) as stream:
                    # Discard buffered audio from speaker output
                    warmup_end = time.time() + 2.0
                    while time.time() < warmup_end:
                        stream.read(self.hop_samples)

                    # Fill buffer with real audio so first inference never sees zeros
                    refill_hops = (self.frame_samples // self.hop_samples) + 2
                    for _ in range(refill_hops):
                        chunk, _ = stream.read(self.hop_samples)
                        chunk = chunk[:, 0]
                        self._buffer[:-self.hop_samples] = self._buffer[self.hop_samples:]
                        self._buffer[-self.hop_samples:] = chunk

                    miss_count = 0
                    consecutive_count = 0
                    while True:
                        chunk, _ = stream.read(self.hop_samples)
                        chunk = chunk[:, 0]

                        # Slide buffer forward
                        self._buffer[:-self.hop_samples] = self._buffer[self.hop_samples:]
                        self._buffer[-self.hop_samples:] = chunk

                        # Energy gate — skip silence
                        rms = np.sqrt(np.mean(self._buffer ** 2))
                        if rms < self.energy_threshold:
                            miss_count = 0
                            consecutive_count = 0
                            continue

                        # Cooldown gate
                        now = time.time()
                        if now - self._last_detect_time < self.cooldown_seconds:
                            continue

                        # Run inference
                        probs = self._infer(self._buffer)

                        wake_score    = float(probs[self.wakeword_class])
                        other_probs   = np.delete(probs, self.wakeword_class)
                        best_other    = float(np.max(other_probs))
                        predicted_cls = int(np.argmax(probs))
                        margin        = wake_score - best_other

                        print(f"Probs    : {np.round(probs, 3)}  (hey_hali | noise | unknown)")
                        print(f"Wake     : {wake_score:.3f}  best_other: {best_other:.3f}  margin: {margin:.3f}  RMS: {rms:.4f}  consec: {consecutive_count}")
                        print("----")

                        # GC + rest after every inference
                        gc.collect()
                        time.sleep(0.25)

                        # First window needs threshold + margin (strong signal)
                        # Second window just needs threshold (confirms word still present)
                        above_threshold = (
                            predicted_cls == self.wakeword_class
                            and wake_score >= self.confidence_threshold
                        )
                        if consecutive_count == 0:
                            if above_threshold and margin >= self.confidence_margin:
                                consecutive_count = 1
                                print("🔔 Candidate (1/2)...")
                            else:
                                consecutive_count = 0
                        else:
                            if above_threshold:
                                print("✅ Wake word detected!")
                                self._last_detect_time = time.time()
                                consecutive_count = 0
                                return True
                            else:
                                consecutive_count = 0

                        # NOISE = model is confident nothing is happening, reset like silence
                        # UNKNOWN = ambiguous speech, count as a miss
                        if predicted_cls == 1:  # noise class
                            miss_count = 0
                            continue

                        # On repeated unknown misses, rest then properly refill buffer
                        miss_count += 1
                        if miss_count >= self.max_misses:
                            print(f"😴 {self.max_misses} misses — resting {self.rest_seconds:.0f}s...\n")
                            miss_count = 0
                            gc.collect()
                            time.sleep(self.rest_seconds)
                            # Drain audio that accumulated in the device buffer during sleep
                            drain_hops = int(self.rest_seconds / self.hop_duration) + 5
                            for _ in range(drain_hops):
                                stream.read(self.hop_samples)
                            # Now fill buffer with fresh audio
                            refill_hops = (self.frame_samples // self.hop_samples) + 2
                            for _ in range(refill_hops):
                                chunk, _ = stream.read(self.hop_samples)
                                chunk = chunk[:, 0]
                                self._buffer[:-self.hop_samples] = self._buffer[self.hop_samples:]
                                self._buffer[-self.hop_samples:] = chunk
                            consecutive_count = 0
                            print("🎧 Listening for wake word...")

            except Exception as e:
                print(f"⚠️  Audio stream error: {e} — restarting listener in 1s...")
                time.sleep(1.0)

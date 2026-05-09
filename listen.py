import sounddevice as sd
import wavio
import numpy as np


def record_audio(filename="input.wav",
                 samplerate=16000,
                 silence_duration=2.0,
                 max_duration=30.0):
    """
    Record until 1 second of silence after speech, or max_duration seconds.
    Calibrates silence threshold from the ambient noise floor at the start.
    Returns (filename, rms_level).
    """
    chunk_duration = 0.1  # 100ms chunks
    chunk_samples  = int(chunk_duration * samplerate)
    silence_needed = int(silence_duration / chunk_duration)
    max_chunks     = int(max_duration / chunk_duration)
    calibrate_chunks = 5  # 0.5s of ambient sampling

    print("Listening... speak now.")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                        blocksize=chunk_samples) as stream:

        # Measure ambient noise floor
        noise_samples = []
        for _ in range(calibrate_chunks):
            chunk, _ = stream.read(chunk_samples)
            noise_samples.append(float(np.sqrt(np.mean(chunk[:, 0] ** 2))))
        noise_floor = float(np.mean(noise_samples))
        # Speech threshold: 3x the noise floor, minimum 0.005
        speech_threshold = max(noise_floor * 3.0, 0.005)
        print(f"  noise floor: {noise_floor:.5f}  speech threshold: {speech_threshold:.5f}")

        frames         = []
        silence_count  = 0
        speech_started = False

        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk_mono = chunk[:, 0]
            rms = float(np.sqrt(np.mean(chunk_mono ** 2)))

            frames.append(chunk_mono)

            if rms > speech_threshold:
                speech_started = True
                silence_count  = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_needed:
                    break

    audio = np.concatenate(frames) if frames else np.zeros(chunk_samples, dtype=np.float32)
    rms   = float(np.sqrt(np.mean(audio ** 2)))
    wavio.write(filename, audio.reshape(-1, 1), samplerate, sampwidth=2)

    return filename, rms

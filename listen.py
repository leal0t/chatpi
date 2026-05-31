import subprocess
import numpy as np
import wave
import io


def record_audio(filename="input.wav",
                 samplerate=16000,
                 silence_duration=1.2,
                 max_duration=30.0):
    """
    Record until 1 second of silence after speech, or max_duration seconds.
    Uses parec (PulseAudio native) to capture from the default source.
    Returns (filename, rms_level).
    """
    chunk_duration   = 0.1
    chunk_samples    = int(chunk_duration * samplerate)
    chunk_bytes      = chunk_samples * 2  # s16le = 2 bytes per sample
    silence_needed   = int(silence_duration / chunk_duration)
    max_chunks       = int(max_duration / chunk_duration)
    no_speech_chunks = int(8.0 / chunk_duration)
    calibrate_chunks = 5

    print("Listening... speak now.")

    cmd = [
        "parec",
        "--rate=16000",
        "--channels=1",
        "--format=s16le",
        "--latency-msec=50",
        "--device=alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        # Calibrate noise floor
        noise_samples = []
        for _ in range(calibrate_chunks):
            raw = proc.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            noise_samples.append(float(np.sqrt(np.mean(chunk ** 2))))
        noise_floor = float(np.mean(noise_samples)) if noise_samples else 0.005
        speech_threshold = max(noise_floor * 2.5, 0.005)
        print(f"  noise floor: {noise_floor:.5f}  speech threshold: {speech_threshold:.5f}")

        frames        = []
        silence_count = 0
        speech_started = False

        for i in range(max_chunks):
            raw = proc.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            frames.append(chunk)

            if rms > speech_threshold:
                speech_started = True
                silence_count  = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_needed:
                    break
            elif not speech_started and i >= no_speech_chunks:
                break

    finally:
        proc.terminate()
        proc.wait()

    audio = np.concatenate(frames) if frames else np.zeros(chunk_samples, dtype=np.float32)
    rms   = float(np.sqrt(np.mean(audio ** 2)))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    return filename, rms

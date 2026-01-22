import time
from wakeword import WakeWordDetector
from listen import record_audio
from chat import transcribe, ask_chatgpt
from speak import speak_audio

WAKE_GREETING = "Hey, what's up. If I've not met you before, my name is Hali. How's everything going?"
SILENCE_TIMEOUT_SECONDS = 10
RECORD_SECONDS = 10  # how long to listen for each turn
SILENCE_RMS_THRESHOLD = 0.005  # tweak this if needed

def conversation_loop():
    """
    Conversation loop that stays awake until there's no *real* speech
    (low RMS audio) for SILENCE_TIMEOUT_SECONDS after Hali's last response.
    """
    print("🗣 Hali is awake. You don't need the wake word now.")
    print("Just talk. If you're quiet for a while, I'll go back to sleep.\n")

    # Say the greeting out loud and start the activity timer
    speak_audio(WAKE_GREETING)
    last_activity = time.monotonic()  # time of last meaningful interaction

    while True:
        print("🎙 Listening for your question (or I'll nap soon)...")
        audio_file, rms = record_audio(RECORD_SECONDS)

        now = time.monotonic()

        # --- First layer: amplitude-based silence detection ---
        if rms < SILENCE_RMS_THRESHOLD:
            print(f"(Silence / very quiet chunk: rms={rms:.4f})")
            # Check if we've been quiet for too long
            if now - last_activity > SILENCE_TIMEOUT_SECONDS:
                print("😴 No speech detected for a bit — going back to sleep.\n")
                return  # exit to main loop, which will wait for wake word again
            else:
                print("🤔 Didn't hear much that time, but I'm still awake.\n")
                # Don't update last_activity, so the silence timer keeps running
                continue

        # --- If it's loud enough, then transcribe ---
        text = transcribe(audio_file)
        print(f"You said: {text!r}")

        # Second layer: if transcription is empty despite sound, treat as silence
        if not text.strip():
            print("🤔 I heard something, but couldn't understand.")
            if now - last_activity > SILENCE_TIMEOUT_SECONDS:
                print("😴 No clear speech for a bit — going back to sleep.\n")
                return
            else:
                continue

        # At this point we heard something meaningful: user spoke
        # Get Hali's response
        response = ask_chatgpt(text)
        print(f"Hali: {response}\n")

        # Speak it out loud
        speak_audio(response)

        # Reset activity timer to "just now" since Hali responded
        last_activity = time.monotonic()


def main():
    wake = WakeWordDetector(threshold=0.5)

    print("Hali is ready on the Pi Zero!")
    print("Say 'Hey Rhasspy' to wake her up. Ctrl+C to quit.\n")

    while True:
        try:
            # 1) Wait for the wake word
            wake.wait_for_wake_word()

            # 2) After wake word, go into a conversation loop
            conversation_loop()

            # When conversation_loop returns, we go back to waiting
            # for "Hey Rhasspy" again.

        except KeyboardInterrupt:
            print("\n👋 Exiting Hali. Goodbye!")
            break


if __name__ == "__main__":
    main()

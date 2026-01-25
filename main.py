import time
import re

from wakeword import WakeWordDetector
from listen import record_audio
from chat import transcribe, ask_chatgpt
from speak import speak_audio

WAKE_GREETING = "Hey, what's up. If I've not met you before, my name is Hali. How's everything going?"
WAKE_SHORT_ACK = "Yeah?"
SLEEP_ACK = "Okay. Going back to sleep."

SILENCE_TIMEOUT_SECONDS = 10
RECORD_SECONDS = 10  # how long to listen for each turn
SILENCE_RMS_THRESHOLD = 0.005  # tweak this if needed

# Phrases that end the session immediately.
GOODBYE_PATTERNS = [
    r"\bgoodbye\b",
    r"\bbye\b",
    r"\bgo to sleep\b",
    r"\bsleep\b",
    r"\bstop listening\b",
    r"\bcancel\b",
    r"\bthat's all\b",
    r"\bthat is all\b",
    r"\bthanks\b.*\bbye\b",
]

def is_goodbye(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in GOODBYE_PATTERNS)

def conversation_loop(say_full_greeting: bool = True):
    """
    Conversation loop that stays awake until:
      - user says a goodbye phrase (immediate sleep), OR
      - there's no *real* speech (low RMS audio) for SILENCE_TIMEOUT_SECONDS
        after Hali's last response.
    """
    print("🗣 Hali is awake. You don't need the wake word now.")
    print("Just talk. Say 'goodbye' (or 'go to sleep') to end.\n")

    # Say greeting out loud and start the activity timer
    if say_full_greeting:
        speak_audio(WAKE_GREETING)
    else:
        speak_audio(WAKE_SHORT_ACK)

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
                return
            else:
                print("🤔 Didn't hear much that time, but I'm still awake.\n")
                # Don't update last_activity, so the silence timer keeps running
                continue

        # --- If it's loud enough, then transcribe ---
        text = transcribe(audio_file)
        print(f"You said: {text!r}")

        clean_text = text.strip().lower()

        # Treat very short or meaningless text as noise
        if not clean_text or len(clean_text) < 3:
            print("🤔 That sounded like noise, not real speech.")
            if now - last_activity > SILENCE_TIMEOUT_SECONDS:
                print("😴 No real speech for a bit — going back to sleep.\n")
                return
            else:
                continue

        # ✅ Goodbye / sleep command
        if is_goodbye(text):
            print("👋 Goodbye phrase detected — sleeping.\n")
            speak_audio(SLEEP_ACK)
            return

        # Heard something meaningful: user spoke
        response = ask_chatgpt(text)
        print(f"Hali: {response}\n")

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
            conversation_loop(say_full_greeting=True)

            # ✅ IMPORTANT: ignore wakeword briefly so we don't re-trigger immediately
            # from buffered mic audio or speaker echo.
            time.sleep(2.0)

            # then we naturally go back to waiting for wake word again

        except KeyboardInterrupt:
            print("\n👋 Exiting Hali. Goodbye!")
            break


if __name__ == "__main__":
    main()

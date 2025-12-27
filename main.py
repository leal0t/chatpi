from wakeword import WakeWordDetector
from listen import record_audio
from chat import transcribe, ask_chatgpt
from speak import speak_audio

wake = WakeWordDetector(threshold=0.5)

print("Hali is ready!")
print("Say 'Hey Rhasspy' to wake her up. Ctrl+C to quit.\n")

while True:
    try:
        wake.wait_for_wake_word()

        print("🔔 Wake word heard. Listening for your question...")
        audio_file = record_audio()

        text = transcribe(audio_file)
        print(f"You said: {text!r}")

        if not text.strip():
            print("I didn't catch that. Let's try again.\n")
            continue

        response = ask_chatgpt(text)
        print(f"Hali: {response}\n")

        speak_audio(response)

    except KeyboardInterrupt:
        print("\n👋 Exiting Hali. Goodbye!")
        break

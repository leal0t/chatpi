from listen import record_audio
from chat import transcribe, ask_chatgpt
from speak import speak_audio

print("Maple assistant is ready.")
print("Press Enter to talk. Press Ctrl+C to quit.\n")

while True:
    try:
        input("👉 Press Enter, then speak when prompted...")
        
        # Record your voice
        audio_file = record_audio()
        
        # Transcribe speech to text
        text = transcribe(audio_file)
        print(f"You said: {text!r}")
        
        if not text.strip():
            print("I didn't catch anything. Let's try again.\n")
            continue
        
        # Ask ChatGPT
        response = ask_chatgpt(text)
        print(f"Maple: {response}\n")
        
        # Speak the response out loud
        speak_audio(response)

    except KeyboardInterrupt:
        print("\nExiting Maple. Goodbye!")
        break

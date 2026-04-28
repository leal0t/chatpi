import time
import re

from edge_wakeword import EdgeWakeWordDetector
from listen import record_audio
from chat import transcribe, ask_chatgpt
from speak import speak_audio

WAKE_GREETING = "Hey, what's up.  How's everything going?"
WAKE_SHORT_ACK = "Yeah?"
SLEEP_ACK = "Okay. Going back to sleep.  me! me! me! me!"
GREET_ACK = "Taking it easy. Just relaxing and enjoying a nice glass of whisky."

SILENCE_TIMEOUT_SECONDS = 10
RECORD_SECONDS = 6
SILENCE_RMS_THRESHOLD = 0.005
MAX_CONVERSATION_TURNS = 12

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

GREETING_PATTERNS = [
	r"\bhow's it going\b",
	r"\bhow are you\b",
	r"\bhow are you doing\b",
	r"\bwhat's up\b",
	r"\bwhats up\b",
	r"\bhey\b",
	r"\bhello\b",
	r"\bhi\b",
	r"\bhey hali\b",
	r"\bhi hali\b",
	r"\bhello hali\b",
	r"\bhali what's up\b",
	r"\bhali how's it going\b",
	r"\bhali how are you\b",
	r"\bwhat are you up to\b",
	r"\bwhat's going on\b",
	r"\bwhat have you been up to\b",
	r"\bhow's your day\b",
	r"\bhow's your day going\b",
	r"\bit's going good\b",
	r"\bit's going well\b",
	r"\bi'm good\b",
	r"\bdoing good\b",
	r"\bdoing well\b",
	r"\bit's going good.*how are you\b",
	r"\bit's going good.*how's it going\b",
	r"\bi'm good.*how about you\b",
	r"\bdoing good.*what about you\b",
]

IGNORE_PHRASES = [
	"thank you for watching",
	"thanks for watching",
	"like and subscribe",
	"subscribe",
	"hit the like button",
	"notification bell",
	"we'll see you next time",
]


def is_goodbye(text: str) -> bool:
	t = (text or "").strip().lower()
	return any(re.search(p, t) for p in GOODBYE_PATTERNS)


def is_greeting(text: str) -> bool:
	t = (text or "").lower()
	GREETING_WORDS = [
		"hello", "hi", "hey",
		"how are you", "how's it going",
		"what's up", "whats up",
		"how's your day",
	]
	return any(w in t for w in GREETING_WORDS)


def conversation_loop(say_full_greeting: bool = True):
	conversation_history = []
	first_turn = True

	print("🗣 Hali is awake. You don't need the wake word now.")
	print("Just talk. Say 'goodbye' (or 'go to sleep') to end.\n")

	if say_full_greeting:
		speak_audio(WAKE_GREETING)
	else:
		speak_audio(WAKE_SHORT_ACK)

	last_activity = time.monotonic()

	while True:
		print("🎙 Listening for your question (or I'll nap soon)...")
		audio_file, rms = record_audio(RECORD_SECONDS)

		now = time.monotonic()

		if rms < SILENCE_RMS_THRESHOLD:
			print(f"(Silence / very quiet chunk: rms={rms:.4f})")
			if now - last_activity > SILENCE_TIMEOUT_SECONDS:
				print("😴 No speech detected for a bit — going back to sleep.\n")
				return
			else:
				print("🤔 Didn't hear much that time, but I'm still awake.\n")
				continue

		text = transcribe(audio_file)
		print(f"You said: {text!r}")

		conversation_history.append({"role": "user", "content": text})
		conversation_history = conversation_history[-MAX_CONVERSATION_TURNS:]

		clean_text = text.strip().lower()

		if any(p in clean_text for p in IGNORE_PHRASES):
			print(f"📺 Ignoring background phrase: {clean_text!r}")
			if now - last_activity > SILENCE_TIMEOUT_SECONDS:
				print("😴 Background noise only — going back to sleep.\n")
				return
			continue

		if not clean_text or len(clean_text) < 3:
			print("🤔 That sounded like noise, not real speech.")
			if now - last_activity > SILENCE_TIMEOUT_SECONDS:
				print("😴 No real speech for a bit — going back to sleep.\n")
				return
			else:
				continue

		if first_turn and is_greeting(text):
			print("👋 Greeting phrase detected (first turn).")
			speak_audio(GREET_ACK)
			last_activity = time.monotonic()
			first_turn = False
			continue

		if is_goodbye(text):
			print("👋 Goodbye phrase detected — sleeping.\n")
			speak_audio(SLEEP_ACK)
			return

		first_turn = False

		response = ask_chatgpt(conversation_history)
		print(f"Hali: {response}\n")

		conversation_history.append({"role": "assistant", "content": response})
		conversation_history = conversation_history[-MAX_CONVERSATION_TURNS:]

		speak_audio(response)
		last_activity = time.monotonic()


def main():
	wake = EdgeWakeWordDetector(
		model_path="/home/pi/chatpi/edge_wakeword/hey_hali.tflite",
		samplerate=16000,
		frame_duration=1.0,
		hop_duration=0.20,
		energy_threshold=0.012,
		wakeword_class=0,
		cooldown_seconds=2.0,
		confidence_threshold=0.40,
		confidence_margin=0.10,
		max_misses=10,
		device=None,
	)

	print("Hali is ready on the Pi Zero!")
	print("Say 'Hey Hali' to wake her up. Ctrl+C to quit.\n")

	try:
		while True:
			detected = wake.wait_for_wake_word()
			if detected:
				conversation_loop(say_full_greeting=True)
				time.sleep(4.0)
			# If not detected (max misses hit), loop back and try again
	except KeyboardInterrupt:
		print("\n👋 Exiting Hali. Goodbye!")


if __name__ == "__main__":
	main()

import gc
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
SILENCE_RMS_THRESHOLD = 0.005
MAX_CONVERSATION_TURNS = 12

GOODBYE_PATTERNS = [
	r"\bgoodbye\b",
	r"\bbye\b",
	r"\bgo to sleep\b",
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
	if re.search(r"\bgo to sleep\b", t):
		return True
	if len(t.split()) > 10:
		return False
	return any(re.search(p, t) for p in GOODBYE_PATTERNS)


def is_greeting(text: str) -> bool:
	t = (text or "").lower()
	if len(t.split()) > 8:
		return False
	GREETING_WORDS = [
		"hello", "hi", "hey",
		"how are you", "how's it going",
		"what's up", "whats up",
		"how's your day",
	]
	return any(w in t for w in GREETING_WORDS)


def conversation_loop(lcd=None, say_full_greeting: bool = True, stop_event=None):
	conversation_history = []
	first_turn = True

	print("🗣 Hali is awake. You don't need the wake word now.")
	print("Just talk. Say 'goodbye' (or 'go to sleep') to end.\n")

	if lcd:
		lcd.show_face(mouth_open=False)

	if say_full_greeting:
		speak_audio(WAKE_GREETING, lcd=lcd)
	else:
		speak_audio(WAKE_SHORT_ACK, lcd=lcd)

	if lcd:
		lcd.show_face(mouth_open=False)

	last_activity = time.monotonic()

	while True:
		if stop_event and stop_event.is_set():
			stop_event.clear()
			print("🔴 Stopped mid-conversation.")
			return

		print("🎙 Listening for your question (or I'll nap soon)...")
		audio_file, rms = record_audio()

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
			speak_audio(GREET_ACK, lcd=lcd)
			if lcd:
				lcd.show_face(mouth_open=False)
			last_activity = time.monotonic()
			first_turn = False
			continue

		if is_goodbye(text):
			print("👋 Goodbye phrase detected — sleeping.\n")
			speak_audio(SLEEP_ACK, lcd=lcd)
			return

		first_turn = False

		response = ask_chatgpt(conversation_history)
		print(f"Hali: {response}\n")

		conversation_history.append({"role": "assistant", "content": response})
		conversation_history = conversation_history[-MAX_CONVERSATION_TURNS:]

		speak_audio(response, lcd=lcd)
		if lcd:
			lcd.show_face(mouth_open=False)
		last_activity = time.monotonic()


def main():
	import threading
	start_event = threading.Event()  # set by KEY1
	stop_event  = threading.Event()  # set by KEY2

	# Try to initialise the LCD.  If hardware isn't present we run headless.
	lcd  = None
	wake = None
	try:
		from lcd import LCDDisplay

		def on_start():
			print("🟢 KEY1: start")
			start_event.set()

		def on_stop():
			print("🔴 KEY2: stop")
			stop_event.set()
			if wake is not None:
				wake.request_stop()

		def on_wake():
			print("⌨️  KEY3: manual wake")
			if wake is not None:
				wake.manual_wake()

		lcd = LCDDisplay(on_start=on_start, on_stop=on_stop, on_wake=on_wake)
		print("🖥  LCD initialised")
	except Exception as e:
		print(f"⚠️  LCD not available ({e}) — running headless")

	wake = EdgeWakeWordDetector(
		model_path="/home/pi/chatpi/edge_wakeword/hey_hali.tflite",
		samplerate=16000,
		frame_duration=1.0,
		hop_duration=0.20,
		energy_threshold=0.020,
		wakeword_class=0,
		cooldown_seconds=2.0,
		confidence_threshold=0.30,
		confidence_margin=0.08,
		max_misses=10,
		rest_seconds=5.0,
		device=None,
		lcd=lcd,
	)

	print("Hali is ready on the Pi 4!")
	print("KEY1=start  KEY2=stop  KEY3=manual wake  Ctrl+C=quit\n")

	# ── Wait for KEY1 before doing anything ───────────────────────────────
	if lcd:
		lcd.show_status("SLEEPING", detail="Press KEY1 to start")
	print("Press KEY1 to start Hali...")
	start_event.wait()
	start_event.clear()
	print("Starting...\n")

	try:
		while True:
			detected = wake.wait_for_wake_word()

			if not detected:
				# KEY2 stop — go dark and wait for KEY1 or KEY3
				print("⏹  Stopped. Press KEY1 or KEY3 to restart.")
				if lcd:
					lcd.show_status("SLEEPING", detail="KEY1=restart  KEY3=wake")
				while not start_event.is_set() and not wake._manual_wake:
					time.sleep(0.1)
				start_event.clear()
				print("Restarting...\n")
				continue

			try:
				conversation_loop(lcd=lcd, say_full_greeting=True, stop_event=stop_event)
			except Exception as e:
				print(f"⚠️  Conversation error: {e}")

			if lcd:
				lcd.show_status("LISTENING")
			gc.collect()
			time.sleep(4.0)

	except KeyboardInterrupt:
		print("\n👋 Exiting Hali. Goodbye!")
	finally:
		if lcd:
			lcd.close()


if __name__ == "__main__":
	main()

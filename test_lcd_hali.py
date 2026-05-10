"""Quick test of lcd.py — run before main.py to check the Hali UI."""
import time
from lcd import LCDDisplay

print("Initialising LCD...")
lcd = LCDDisplay()
print("LCD init OK")

print("Status: READY")
lcd.show_status("READY")
time.sleep(2)

print("Status: LISTENING  0.123")
lcd.show_status("LISTENING", score=0.123)
time.sleep(2)

print("Status: CANDIDATE  0.456")
lcd.show_status("CANDIDATE", score=0.456)
time.sleep(2)

print("Status: MISS  0.089")
lcd.show_status("MISS", score=0.089)
time.sleep(2)

print("Face: mouth closed (blinking)")
lcd.show_face(mouth_open=False)
time.sleep(4)

print("Face: mouth open (talking)")
lcd.show_face(mouth_open=True)
time.sleep(2)

print("Face: mouth closed again")
lcd.show_face(mouth_open=False)
time.sleep(2)

print("Done — closing LCD")
lcd.close()

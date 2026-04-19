#!/usr/bin/env python3
"""
button_handler.py — USB button controller for Hali

Button 1 (CTRL+C)     → Start Hali (python3 main.py)
Button 2 (CTRL+V)     → Stop Hali (kill main.py)
Both together (SHIFT+C+V) → Trigger wake word instantly
"""

import os
import signal
import subprocess
import time
from evdev import InputDevice, categorize, ecodes

DEVICE      = "/dev/input/event4"
HALI_SCRIPT = "/home/pi/chatpi/main.py"
WAKE_FLAG   = "/tmp/hali_wake_trigger"

# Key codes from evtest
KEY_C      = 46
KEY_V      = 47
KEY_LCTRL  = 29
KEY_LSHIFT = 42

hali_process = None
pressed_keys = set()


def start_hali():
    global hali_process
    if hali_process and hali_process.poll() is None:
        print("⚠️  Hali is already running")
        return
    print("▶️  Starting Hali...")
    hali_process = subprocess.Popen(
        ["python3", HALI_SCRIPT],
        cwd="/home/pi/chatpi",
    )
    print(f"✅ Hali started (PID {hali_process.pid})")


def stop_hali():
    global hali_process
    if hali_process is None or hali_process.poll() is not None:
        print("⚠️  Hali is not running")
        return
    print("⏹️  Stopping Hali...")
    hali_process.terminate()
    try:
        hali_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        hali_process.kill()
    hali_process = None
    print("✅ Hali stopped")


def trigger_wake():
    if hali_process is None or hali_process.poll() is not None:
        print("⚠️  Hali isn't running — starting her first...")
        start_hali()
        time.sleep(3.0)  # give her time to boot up
    print("🔔 Triggering wake word...")
    # Write a flag file that main.py watches to skip wake word detection
    with open(WAKE_FLAG, "w") as f:
        f.write("wake")


def handle_keys():
    global pressed_keys

    # Track which combo we just fired so we don't repeat
    combo_fired = False

    device = InputDevice(DEVICE)
    print(f"🎮 Button handler listening on {DEVICE}")
    print("   Button 1 (CTRL+C)     → Start Hali")
    print("   Button 2 (CTRL+V)     → Stop Hali")
    print("   Both buttons (SHIFT+C+V) → Trigger wake\n")

    for event in device.read_loop():
        if event.type != ecodes.EV_KEY:
            continue

        key_event = categorize(event)

        # Track key down (value 1) and key up (value 0)
        if key_event.keystate == 1:  # key down
            pressed_keys.add(key_event.scancode)
        elif key_event.keystate == 0:  # key up
            pressed_keys.discard(key_event.scancode)
            combo_fired = False  # reset so next press can fire

        # Both buttons together: SHIFT + C + V
        if (KEY_LSHIFT in pressed_keys
                and KEY_C in pressed_keys
                and KEY_V in pressed_keys
                and not combo_fired):
            combo_fired = True
            trigger_wake()

        # Button 1 alone: CTRL + C (no SHIFT, no V)
        elif (KEY_LCTRL in pressed_keys
                and KEY_C in pressed_keys
                and KEY_V not in pressed_keys
                and KEY_LSHIFT not in pressed_keys
                and not combo_fired):
            combo_fired = True
            start_hali()

        # Button 2 alone: CTRL + V (no SHIFT, no C)
        elif (KEY_LCTRL in pressed_keys
                and KEY_V in pressed_keys
                and KEY_C not in pressed_keys
                and KEY_LSHIFT not in pressed_keys
                and not combo_fired):
            combo_fired = True
            stop_hali()


if __name__ == "__main__":
    try:
        handle_keys()
    except KeyboardInterrupt:
        print("\n👋 Button handler exiting")
        stop_hali()

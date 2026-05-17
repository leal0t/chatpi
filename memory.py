import json
import os

MEMORY_FILE = "/home/pi/chatpi/memory.json"
MAX_MESSAGES = 40  # 20 exchanges

def load_history() -> list[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return data[-MAX_MESSAGES:] if isinstance(data, list) else []
    except Exception:
        return []

def save_history(history: list[dict]):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history[-MAX_MESSAGES:], f)
    except Exception as e:
        print(f"[memory] Failed to save: {e}")

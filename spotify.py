import os
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
_REFRESH_TOKEN = os.getenv("SPOTIFY_REFRESH_TOKEN")

_BASE   = "https://api.spotify.com/v1"
_TOKEN_URL = "https://accounts.spotify.com/api/token"

_access_token    = None
_token_expires_at = 0.0


def _get_token() -> str | None:
    global _access_token, _token_expires_at
    if not _CLIENT_ID or not _CLIENT_SECRET or not _REFRESH_TOKEN:
        return None
    if _access_token and time.time() < _token_expires_at:
        return _access_token
    creds = base64.b64encode(f"{_CLIENT_ID}:{_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(_TOKEN_URL,
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "refresh_token", "refresh_token": _REFRESH_TOKEN},
        timeout=10)
    if not resp.ok:
        print(f"[spotify] Token refresh failed: {resp.text}")
        return None
    data = resp.json()
    _access_token     = data["access_token"]
    _token_expires_at = time.time() + data.get("expires_in", 3600) - 60
    return _access_token


def _headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}"}


def _active_device() -> str | None:
    token = _get_token()
    if not token:
        return None
    resp = requests.get(f"{_BASE}/me/player/devices", headers=_headers(), timeout=10)
    if not resp.ok:
        return None
    devices = resp.json().get("devices", [])
    active = next((d for d in devices if d.get("is_active")), None)
    if active:
        return active["id"]
    return devices[0]["id"] if devices else None


def is_configured() -> bool:
    return bool(_CLIENT_ID and _CLIENT_SECRET and _REFRESH_TOKEN)


def play_search(query: str) -> bool:
    """Search Spotify and play the top playlist or track match."""
    token = _get_token()
    if not token:
        return False
    resp = requests.get(f"{_BASE}/search", headers=_headers(),
        params={"q": query, "type": "playlist,track", "limit": 1}, timeout=10)
    if not resp.ok:
        return False
    data = resp.json()
    playlists = data.get("playlists", {}).get("items", [])
    tracks    = data.get("tracks",    {}).get("items", [])
    device_id = _active_device()
    if not device_id:
        print("[spotify] No active device found")
        return False
    if playlists:
        body = {"context_uri": playlists[0]["uri"], "offset": {"position": 0}}
    elif tracks:
        body = {"uris": [tracks[0]["uri"]]}
    else:
        return False
    r = requests.put(f"{_BASE}/me/player/play?device_id={device_id}",
        headers={**_headers(), "Content-Type": "application/json"},
        json=body, timeout=10)
    return r.status_code in (200, 204)


def pause() -> bool:
    device_id = _active_device()
    params = {"device_id": device_id} if device_id else {}
    r = requests.put(f"{_BASE}/me/player/pause", headers=_headers(), params=params, timeout=10)
    return r.status_code in (200, 204)


def resume() -> bool:
    device_id = _active_device()
    params = {"device_id": device_id} if device_id else {}
    r = requests.put(f"{_BASE}/me/player/play", headers=_headers(), params=params, timeout=10)
    return r.status_code in (200, 204)


def skip() -> bool:
    r = requests.post(f"{_BASE}/me/player/next", headers=_headers(), timeout=10)
    return r.status_code in (200, 204)


def set_volume(percent: int) -> bool:
    percent   = max(0, min(100, int(percent)))
    device_id = _active_device()
    params    = {"volume_percent": percent}
    if device_id:
        params["device_id"] = device_id
    r = requests.put(f"{_BASE}/me/player/volume", headers=_headers(), params=params, timeout=10)
    return r.status_code in (200, 204)


def now_playing() -> dict | None:
    token = _get_token()
    if not token:
        return None
    r = requests.get(f"{_BASE}/me/player/currently-playing", headers=_headers(), timeout=10)
    if r.status_code == 204 or not r.ok:
        return None
    data = r.json()
    item = data.get("item")
    if not item:
        return None
    return {
        "name":   item["name"],
        "artist": item["artists"][0]["name"],
        "album":  item["album"]["name"],
    }

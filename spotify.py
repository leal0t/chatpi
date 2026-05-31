import os
import time
import base64
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

_CLIENT_ID     = "SPOTIFY_CLIENT_ID_REMOVED"
_CLIENT_SECRET = "SPOTIFY_CLIENT_SECRET_REMOVED"
_REFRESH_TOKEN = "SPOTIFY_REFRESH_TOKEN_REMOVED"

DEVICE_MAP = {
    "echo":  "Karen's Echo Dot",
    "cinc":  "CINC-FKZRKG3",
}

_BASE      = "https://api.spotify.com/v1"
_TOKEN_URL = "https://accounts.spotify.com/api/token"

_access_token     = None
_token_expires_at = 0.0

_playback_thread = None
_stop_playback   = threading.Event()


# ── Auth ──────────────────────────────────────────────────────────────────────

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


# ── Device helpers ────────────────────────────────────────────────────────────

def _active_device(hint: str = None) -> str | None:
    if not _get_token():
        return None
    resp = requests.get(f"{_BASE}/me/player/devices", headers=_headers(), timeout=10)
    if not resp.ok:
        return None
    devices = resp.json().get("devices", [])
    if not devices:
        return None
    if hint:
        target = DEVICE_MAP.get(hint.strip().lower(), hint)
        match = next((d for d in devices if target.lower() in d.get("name", "").lower()), None)
        if match:
            return match["id"]
    active = next((d for d in devices if d.get("is_active")), None)
    return active["id"] if active else devices[0]["id"]


# ── Playlist helpers ──────────────────────────────────────────────────────────

def _get_all_playlist_tracks(playlist_id: str) -> list[dict]:
    tracks = []
    url = f"{_BASE}/playlists/{playlist_id}/tracks?limit=100"
    while url:
        r = requests.get(url, headers=_headers(), timeout=10)
        if not r.ok:
            break
        data = r.json()
        for item in data.get("items", []):
            track = item.get("track") if item else None
            if track and track.get("uri"):
                tracks.append({"uri": track["uri"], "name": track.get("name", "")})
        url = data.get("next")
    return tracks


def _remove_track(playlist_id: str, track_uri: str) -> bool:
    r = requests.delete(f"{_BASE}/playlists/{playlist_id}/tracks",
        headers={**_headers(), "Content-Type": "application/json"},
        json={"tracks": [{"uri": track_uri}]},
        timeout=10)
    return r.ok


def find_temp_playlist(genre: str) -> dict | None:
    """Find first playlist named 'Temp (...)' containing the genre string."""
    genre_lower = genre.strip().lower()
    url = f"{_BASE}/me/playlists?limit=50"
    while url:
        r = requests.get(url, headers=_headers(), timeout=10)
        if not r.ok:
            break
        data = r.json()
        for item in data.get("items", []):
            if not item:
                continue
            name = item.get("name", "")
            if name.startswith("Temp (") and genre_lower in name.lower():
                return item
        url = data.get("next")
    return None


# ── Temp playlist playback (with track deletion) ──────────────────────────────

def _play_uri(uri: str, device_id: str) -> bool:
    return _play_with_retry({"uris": [uri]}, device_id)


def _play_with_retry(body: dict, device_id: str, retries: int = 3, delay: float = 2.5) -> bool:
    for attempt in range(retries):
        r = requests.put(f"{_BASE}/me/player/play?device_id={device_id}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=body, timeout=10)
        if r.status_code in (200, 204):
            return True
        print(f"[spotify] Play attempt {attempt + 1} failed ({r.status_code}), retrying...")
        time.sleep(delay)
    print("[spotify] All play attempts failed")
    return False


def _manage_temp_playlist(playlist_id: str, queue: list, device_hint: str):
    """Background thread: remove tracks from the playlist as Spotify plays/skips past them."""
    remaining = list(queue)
    no_signal_streak = 0

    print(f"[spotify] Managing temp playlist — {len(remaining)} tracks")

    while not _stop_playback.is_set() and remaining:
        time.sleep(6)
        current = now_playing()

        if not current:
            no_signal_streak += 1
            # If nothing is playing for ~18s and only the last track is left, it finished
            if no_signal_streak >= 3 and len(remaining) == 1:
                print("[spotify] Last track finished, removing")
                _remove_track(playlist_id, remaining[0])
                remaining.clear()
            continue

        no_signal_streak = 0
        current_uri = current.get("uri")
        if not current_uri or current_uri not in remaining:
            continue

        idx = remaining.index(current_uri)
        if idx > 0:
            # Remove everything before current track (played or skipped)
            for uri in remaining[:idx]:
                print("[spotify] Removing played/skipped track")
                _remove_track(playlist_id, uri)
            remaining = remaining[idx:]

    print("[spotify] Temp playlist session ended")


def play_temp_playlist(playlist_id: str, device_hint: str = None) -> bool:
    """Play a temp playlist track by track, removing each after it plays."""
    global _playback_thread, _stop_playback

    _stop_playback.set()
    if _playback_thread and _playback_thread.is_alive():
        _playback_thread.join(timeout=3)
    _stop_playback.clear()

    tracks = _get_all_playlist_tracks(playlist_id)
    if not tracks:
        print("[spotify] Temp playlist is empty")
        return False

    device_id = _active_device(device_hint)
    if not device_id:
        print("[spotify] No active device found")
        return False

    # Play playlist as context so Spotify auto-advances between tracks
    playlist_uri = f"spotify:playlist:{playlist_id}"
    if not _play_with_retry({"context_uri": playlist_uri, "offset": {"position": 0}}, device_id):
        print("[spotify] Failed to start playlist")
        return False

    print(f"[spotify] Started temp playlist — {len(tracks)} tracks")

    _playback_thread = threading.Thread(
        target=_manage_temp_playlist,
        args=(playlist_id, [t["uri"] for t in tracks], device_hint),
        daemon=True
    )
    _playback_thread.start()
    return True


# ── Public controls ───────────────────────────────────────────────────────────

def is_configured() -> bool:
    return bool(_CLIENT_ID and _CLIENT_SECRET and _REFRESH_TOKEN)


_RANDOMPLAY_URL = "https://spotifyrandomplay.onrender.com"

def clear_all():
    """Stop Hali's managed session and clear web app sessions."""
    global _playback_thread, _stop_playback
    _stop_playback.set()
    if _playback_thread and _playback_thread.is_alive():
        _playback_thread.join(timeout=3)
    _stop_playback.clear()
    try:
        r = requests.post(f"{_RANDOMPLAY_URL}/playlist/clearallsessions", timeout=5)
        print(f"[spotify] Web sessions cleared: {r.status_code}")
    except Exception as e:
        print(f"[spotify] Could not clear web sessions: {e}")
    print("[spotify] All sessions cleared")


def play_search(query: str, device_hint: str = None) -> bool:
    """Check for a matching Temp playlist first, then fall back to Spotify search."""
    if not _get_token():
        return False

    clear_all()

    temp = find_temp_playlist(query)
    if temp:
        print(f"[spotify] Found temp playlist: {temp['name']}")
        return play_temp_playlist(temp["id"], device_hint)

    resp = requests.get(f"{_BASE}/search", headers=_headers(),
        params={"q": query, "type": "playlist,track", "limit": 1}, timeout=10)
    if not resp.ok:
        return False
    data      = resp.json()
    playlists = [p for p in data.get("playlists", {}).get("items", []) if p]
    tracks    = [t for t in data.get("tracks",    {}).get("items", []) if t]
    device_id = _active_device(device_hint)
    if not device_id:
        print("[spotify] No active device found")
        return False
    if playlists:
        body = {"context_uri": playlists[0]["uri"], "offset": {"position": 0}}
    elif tracks:
        body = {"uris": [tracks[0]["uri"]]}
    else:
        return False
    return _play_with_retry(body, device_id)


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
    if not _get_token():
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
        "uri":    item["uri"],
    }

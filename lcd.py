"""Waveshare 1.44" 128x128 ST7735S LCD HAT driver + Hali UI."""
import time
import threading
import spidev
import RPi.GPIO as GPIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Pin assignments (BCM) ──────────────────────────────────────────────────
RST_PIN  = 27
DC_PIN   = 25
BL_PIN   = 24
KEY1_PIN = 21   # Start Hali
KEY2_PIN = 20   # Stop  Hali
KEY3_PIN = 16   # Manual wake
JOY_UP   = 6    # Start/restart Hali
JOY_DOWN = 19   # Shutdown Pi

WIDTH  = 128
HEIGHT = 128

# ── Palette ────────────────────────────────────────────────────────────────
_SKIN  = (232, 188, 148)
_HAIR  = (38,  16,   3)
_WHITE = (255, 255, 255)
_IRIS  = (92,  58,  22)
_PUPIL = (12,   6,   2)
_LIP   = (192,  68,  68)
_BROW  = (36,  16,   3)
_BG    = (235, 212, 192)

# ── Fonts ──────────────────────────────────────────────────────────────────
def _ttf(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fnt_sm = _ttf(_SANS,  9)
_fnt_md = _ttf(_BOLD, 13)
_fnt_lg = _ttf(_BOLD, 20)


class LCDDisplay:
    """
    Thread-safe LCD driver.  Call show_status() or show_face() from any thread;
    a background thread renders and pushes frames at ~12 fps.

    Button callbacks (on_start / on_stop / on_wake) are called in their own
    daemon threads so they can block without stalling the display.
    """

    def __init__(self, on_start=None, on_stop=None, on_wake=None, on_shutdown=None, on_restart=None):
        self.on_start    = on_start
        self.on_stop     = on_stop
        self.on_wake     = on_wake
        self.on_shutdown = on_shutdown
        self.on_restart  = on_restart

        self._lock    = threading.Lock()
        self._mode    = "status"
        self._status  = {"label": "READY", "score": None, "detail": None}
        self._mouth   = False
        self._running = True

        self._gpio_init()
        self._spi_init()
        self._display_init()
        self._load_face_images()

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self._btn_thread = threading.Thread(target=self._button_loop, daemon=True)
        self._btn_thread.start()

    # ── Hardware init ──────────────────────────────────────────────────────

    def _gpio_init(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin, mode in [(RST_PIN, GPIO.OUT), (DC_PIN, GPIO.OUT), (BL_PIN, GPIO.OUT)]:
            GPIO.setup(pin, mode)
        GPIO.output(BL_PIN, GPIO.HIGH)
        for pin in (KEY1_PIN, KEY2_PIN, KEY3_PIN, JOY_UP, JOY_DOWN):
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _spi_init(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 16_000_000
        self.spi.mode = 0

    def _cmd(self, c):
        GPIO.output(DC_PIN, GPIO.LOW)
        self.spi.writebytes([c])

    def _dat(self, d):
        GPIO.output(DC_PIN, GPIO.HIGH)
        if isinstance(d, int):
            self.spi.writebytes([d])
        else:
            for i in range(0, len(d), 4096):
                self.spi.writebytes(d[i:i + 4096])

    def _display_init(self):
        GPIO.output(RST_PIN, GPIO.HIGH); time.sleep(0.1)
        GPIO.output(RST_PIN, GPIO.LOW);  time.sleep(0.1)
        GPIO.output(RST_PIN, GPIO.HIGH); time.sleep(0.1)
        self._cmd(0x01); time.sleep(0.15)
        self._cmd(0x11); time.sleep(0.50)
        self._cmd(0x3A); self._dat(0x05)   # 16-bit RGB565
        self._cmd(0x36); self._dat(0xC8)   # row/col swap + RGB order
        self._cmd(0x20)                    # inversion off
        self._cmd(0x29); time.sleep(0.10)  # display on

    def _flush(self, img: Image.Image):
        """Convert PIL RGB image → RGB565 big-endian and push to display."""
        img  = img.rotate(90)
        arr  = np.array(img, dtype=np.uint16)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        px   = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        data = px.astype(">u2").tobytes()
        self._cmd(0x2A); self._dat([0x00, 0, 0x00, WIDTH  - 1])
        self._cmd(0x2B); self._dat([0x00, 0, 0x00, HEIGHT - 1])
        self._cmd(0x2C)
        GPIO.output(DC_PIN, GPIO.HIGH)
        for i in range(0, len(data), 4096):
            self.spi.writebytes(list(data[i:i + 4096]))

    # ── Button callbacks ───────────────────────────────────────────────────

    def _cb_start(self, _):
        if self.on_start:
            threading.Thread(target=self.on_start, daemon=True).start()

    def _cb_stop(self, _):
        if self.on_stop:
            threading.Thread(target=self.on_stop, daemon=True).start()

    def _cb_wake(self, _):
        if self.on_wake:
            threading.Thread(target=self.on_wake, daemon=True).start()

    def _cb_shutdown(self, _):
        if self.on_shutdown:
            threading.Thread(target=self.on_shutdown, daemon=True).start()

    def _cb_restart(self, _):
        if self.on_restart:
            threading.Thread(target=self.on_restart, daemon=True).start()

    # ── Public API ─────────────────────────────────────────────────────────

    def show_status(self, label: str, score: float = None, detail: str = None):
        """Display status screen (used during sleep / wake-word listening)."""
        with self._lock:
            self._mode   = "status"
            self._status = {"label": label, "score": score, "detail": detail}

    def show_face(self, mouth_open: bool = False):
        """Display Hali's face (used during active conversation)."""
        with self._lock:
            self._mode  = "face"
            self._mouth = mouth_open

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.spi.close()
        GPIO.cleanup()

    # ── Drawing helpers ────────────────────────────────────────────────────

    def _draw_status(self, label, score, detail) -> Image.Image:
        img  = Image.new("RGB", (WIDTH, HEIGHT), (8, 8, 28))
        draw = ImageDraw.Draw(img)

        # Title bar
        draw.text((64, 6), "HALI", font=_fnt_md, fill=(85, 165, 255), anchor="mt")
        draw.line([(6, 22), (122, 22)], fill=(48, 48, 88))

        # Status word
        color = {
            "SLEEPING":  (65,  65, 130),
            "READY":     (120, 120, 120),
            "LISTENING": (60,  210,  60),
            "CANDIDATE": (255, 205,   0),
            "DETECTED":  (0,   255, 110),
            "MISS":      (160,  60,  60),
        }.get(label.upper(), (200, 200, 200))
        draw.text((64, 58), label.upper(), font=_fnt_lg, fill=color, anchor="mm")

        # Score
        if score is not None:
            draw.text((64, 80), f"{score:.3f}", font=_fnt_sm, fill=(160, 160, 160), anchor="mm")

        # Optional detail line
        if detail:
            draw.text((64, 94), detail, font=_fnt_sm, fill=(110, 110, 110), anchor="mm")

        # Footer
        draw.line([(6, 108), (122, 108)], fill=(48, 48, 88))
        draw.text((64, 119), "1=start  2=stop  3=wake", font=_fnt_sm, fill=(52, 52, 78), anchor="mm")
        return img

    def _load_face_images(self):
        base = "/home/pi/chatpi/"
        size = (WIDTH, HEIGHT)
        try:
            self._img_normal = Image.open(base + "face_base.png").convert("RGB").resize(size, Image.LANCZOS)
            self._img_blink  = Image.open(base + "eyes_closed.png").convert("RGB").resize(size, Image.LANCZOS)
            self._img_talk   = Image.open(base + "mouth_open.png").convert("RGB").resize(size, Image.LANCZOS)
            print("🖼  Face images loaded")
        except FileNotFoundError as e:
            print(f"[LCD] Face image missing ({e}) — using built-in face")
            self._img_normal = self._img_blink = self._img_talk = None

    def _draw_face(self, eyes_open: bool, mouth_open: bool) -> Image.Image:
        if self._img_normal:
            if mouth_open:
                return self._img_talk
            if not eyes_open:
                return self._img_blink
            return self._img_normal

        # Fallback: plain coloured square if images missing
        return Image.new("RGB", (WIDTH, HEIGHT), _BG)

    # ── Button polling thread ──────────────────────────────────────────────

    def _button_loop(self):
        last = {KEY1_PIN: 1, KEY2_PIN: 1, KEY3_PIN: 1, JOY_UP: 1, JOY_DOWN: 1}
        callbacks = {KEY1_PIN: self._cb_start, KEY2_PIN: self._cb_stop, KEY3_PIN: self._cb_wake,
                     JOY_UP: self._cb_restart, JOY_DOWN: self._cb_shutdown}
        while self._running:
            for pin, cb in callbacks.items():
                val = GPIO.input(pin)
                if val == 0 and last[pin] == 1:   # falling edge = press
                    cb(pin)
                last[pin] = val
            time.sleep(0.05)

    # ── Render thread ──────────────────────────────────────────────────────

    def _render_loop(self):
        eyes_open  = True
        last_blink = time.time()
        blink_int  = 3.8   # seconds between blinks
        blink_dur  = 0.12  # how long eyes stay shut

        while self._running:
            try:
                with self._lock:
                    mode   = self._mode
                    status = dict(self._status)
                    mouth  = self._mouth

                if mode == "status":
                    img = self._draw_status(**status)
                else:
                    now = time.time()
                    if eyes_open and now - last_blink >= blink_int:
                        eyes_open  = False
                        last_blink = now
                    elif not eyes_open and now - last_blink >= blink_dur:
                        eyes_open = True

                    img = self._draw_face(eyes_open, mouth)

                self._flush(img)
                time.sleep(0.08)   # ~12 fps

            except Exception as e:
                import traceback
                print(f"[LCD render error] {e}")
                traceback.print_exc()
                time.sleep(1.0)   # pause before retrying

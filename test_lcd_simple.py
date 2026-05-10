"""
Step-by-step LCD HAT test.
Run with: python3 test_lcd_simple.py
"""
import time
import spidev
import RPi.GPIO as GPIO

RST_PIN  = 27
DC_PIN   = 25
BL_PIN   = 24
KEY1_PIN = 21
KEY2_PIN = 20
KEY3_PIN = 16

WIDTH  = 128
HEIGHT = 128


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT)
    GPIO.setup(DC_PIN,  GPIO.OUT)
    GPIO.setup(BL_PIN,  GPIO.OUT)
    for pin in (KEY1_PIN, KEY2_PIN, KEY3_PIN):
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def cmd(spi, c):
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([c])


def dat(spi, d):
    GPIO.output(DC_PIN, GPIO.HIGH)
    if isinstance(d, int):
        spi.writebytes([d])
    else:
        for i in range(0, len(d), 4096):
            spi.writebytes(d[i:i+4096])


def reset():
    GPIO.output(RST_PIN, GPIO.HIGH); time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.LOW);  time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.HIGH); time.sleep(0.1)


def init_display(spi):
    reset()
    cmd(spi, 0x01); time.sleep(0.15)   # software reset
    cmd(spi, 0x11); time.sleep(0.50)   # sleep out
    cmd(spi, 0x3A); dat(spi, 0x05)     # 16-bit RGB565
    cmd(spi, 0x36); dat(spi, 0xC8)     # row/col swap
    cmd(spi, 0x20)                     # inversion off
    cmd(spi, 0x29); time.sleep(0.10)   # display on


def fill(spi, r, g, b):
    color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    hi = (color >> 8) & 0xFF
    lo = color & 0xFF
    cmd(spi, 0x2A); dat(spi, [0x00, 0, 0x00, WIDTH  - 1])
    cmd(spi, 0x2B); dat(spi, [0x00, 0, 0x00, HEIGHT - 1])
    cmd(spi, 0x2C)
    chunk = [hi, lo] * 64
    GPIO.output(DC_PIN, GPIO.HIGH)
    for _ in range(WIDTH * HEIGHT // 64):
        spi.writebytes(chunk)


def main():
    setup()
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 16_000_000
    spi.mode = 0

    try:
        # ── Step 1: backlight only ─────────────────────────────────────
        print("\nStep 1: Backlight ON (should see white glow from screen edge)")
        GPIO.output(BL_PIN, GPIO.HIGH)
        input("Press Enter to continue...")

        # ── Step 2: init + red fill ────────────────────────────────────
        print("\nStep 2: Initialising display and filling RED")
        init_display(spi)
        fill(spi, 255, 0, 0)
        input("See a red screen? Press Enter to continue...")

        # ── Step 3: colour cycle ───────────────────────────────────────
        print("\nStep 3: Colour cycle  GREEN → BLUE → WHITE  (1 s each)")
        for name, r, g, b in [("GREEN", 0, 255, 0), ("BLUE", 0, 0, 255), ("WHITE", 255, 255, 255)]:
            print(f"  → {name}")
            fill(spi, r, g, b)
            time.sleep(1.0)
        input("Saw the colours? Press Enter to continue...")

        # ── Step 4: button test ────────────────────────────────────────
        print("\nStep 4: Button test — press KEY1 / KEY2 / KEY3 (Ctrl+C to quit)")
        print("  KEY1=GPIO21  KEY2=GPIO20  KEY3=GPIO16")
        last = {KEY1_PIN: 1, KEY2_PIN: 1, KEY3_PIN: 1}
        labels = {KEY1_PIN: "KEY1", KEY2_PIN: "KEY2", KEY3_PIN: "KEY3"}
        while True:
            for pin in (KEY1_PIN, KEY2_PIN, KEY3_PIN):
                val = GPIO.input(pin)
                if val == 0 and last[pin] == 1:
                    print(f"  ✅ {labels[pin]} pressed!")
                    fill(spi, *(
                        (255,  80,  80) if pin == KEY1_PIN else
                        ( 80, 255,  80) if pin == KEY2_PIN else
                        ( 80,  80, 255)
                    ))
                last[pin] = val
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        spi.close()
        GPIO.cleanup()


if __name__ == "__main__":
    main()

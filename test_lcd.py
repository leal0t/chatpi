"""
Quick ST7735S 1.44" LCD test.
Fills screen red → green → blue, 1 second each.

Waveshare 1.44" LCD HAT default pinout:
  RST  → GPIO 27  (pin 13)
  DC   → GPIO 25  (pin 22)
  BL   → GPIO 24  (pin 18)  backlight
  CS   → CE0 / GPIO 8  (handled by SPI)
  CLK  → GPIO 11  (handled by SPI)
  MOSI → GPIO 10  (handled by SPI)

If your HAT uses different pins, edit RST_PIN / DC_PIN / BL_PIN below.
"""

import time
import spidev
import RPi.GPIO as GPIO

# --- Pin config (BCM numbering) ---
RST_PIN = 27
DC_PIN  = 25
BL_PIN  = 24

# --- Display dimensions ---
WIDTH  = 128
HEIGHT = 128

def write_cmd(spi, cmd):
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([cmd])

def write_data(spi, data):
    GPIO.output(DC_PIN, GPIO.HIGH)
    if isinstance(data, int):
        spi.writebytes([data])
    else:
        # Send in chunks to avoid SPI buffer limits
        for i in range(0, len(data), 4096):
            spi.writebytes(data[i:i+4096])

def reset(spi):
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.1)

def init_display(spi):
    reset(spi)
    write_cmd(spi, 0x01)   # Software reset
    time.sleep(0.15)
    write_cmd(spi, 0x11)   # Sleep out
    time.sleep(0.5)

    write_cmd(spi, 0x3A)   # Color mode: 16-bit RGB565
    write_data(spi, 0x05)

    write_cmd(spi, 0x36)   # Memory access control
    write_data(spi, 0xC8)  # Row/col swap + RGB order

    write_cmd(spi, 0x20)   # Display inversion off

    write_cmd(spi, 0x29)   # Display on
    time.sleep(0.1)

def set_window(spi, x0, y0, x1, y1):
    write_cmd(spi, 0x2A)   # Column address
    write_data(spi, [0x00, x0, 0x00, x1])
    write_cmd(spi, 0x2B)   # Row address
    write_data(spi, [0x00, y0, 0x00, y1])
    write_cmd(spi, 0x2C)   # Memory write

def fill_color(spi, r, g, b):
    """Fill entire screen with one RGB565 color."""
    # RGB888 → RGB565
    color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    hi = (color >> 8) & 0xFF
    lo = color & 0xFF
    set_window(spi, 0, 0, WIDTH - 1, HEIGHT - 1)
    chunk = [hi, lo] * 64        # 64 pixels per chunk
    write_data(spi, chunk * (WIDTH * HEIGHT // 64))

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT)
    GPIO.setup(DC_PIN,  GPIO.OUT)
    GPIO.setup(BL_PIN,  GPIO.OUT)
    GPIO.output(BL_PIN, GPIO.HIGH)   # backlight on

    spi = spidev.SpiDev()
    spi.open(0, 0)                   # bus 0, CE0
    spi.max_speed_hz = 16_000_000
    spi.mode = 0

    try:
        print("Initialising display...")
        init_display(spi)
        print("Display init done. You should see colours cycling.")

        colours = [
            ("RED",   255, 0,   0),
            ("GREEN", 0,   255, 0),
            ("BLUE",  0,   0,   255),
            ("WHITE", 255, 255, 255),
        ]
        for name, r, g, b in colours:
            print(f"  → {name}")
            fill_color(spi, r, g, b)
            time.sleep(1.5)

        print("\nDone. If you saw 4 solid colours the display is working.")
        print("If the screen stayed black, check your soldering on the SPI/DC/RST pins.")

    finally:
        spi.close()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

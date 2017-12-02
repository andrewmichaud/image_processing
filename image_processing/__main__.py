"""Main class for image processing, for testing mostly."""
#!/usr/bin/env python3

import seamcarve
import shared

if __name__ == "__main__":
    OUT_PIXELS = seamcarve.horizontal_seamcarve(percent=80)
    shared.save_pixels(OUT_PIXELS)

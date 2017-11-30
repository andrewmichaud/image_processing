"""Main class for image processing, for testing mostly."""
#!/usr/bin/env python3

import seamcarve
import shared

if __name__ == "__main__":
    OUT_PIXELS = seamcarve.vertical_seamcarve()
    shared.save_pixels(OUT_PIXELS)

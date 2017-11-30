"""Image processing."""
import itertools

from PIL import Image

import shared

OUTLINE_COLOR = (0, 0, 0, 254)
OUTLINE_CUTOFF = 50

class OutlineData:
    """Data for outlining."""
    def __init__(self, in_name=shared.IN_NAME, out_name=shared.OUT_NAME):
        self.in_image = Image.open(in_name)
        self.height = self.in_image.height
        self.width = self.in_image.width

        self.out_image = Image.new("RGBA", (self.width, self.height), shared.BG_COLOR)
        self.out_name = out_name

        self.in_pixels = self.in_image.load()
        self.out_pixels = self.out_image.load()

        self.shrink_factor = 40

        self.final = Image.new("RGBA", (self.width//self.shrink_factor,
                                        self.height//self.shrink_factor), shared.BG_COLOR)

    def outline(self):
        """Attempt to draw the outline of an image."""
        for y in range(0, self.height, self.shrink_factor):
            for x in range(0, self.width, self.shrink_factor):
                neighbors = shared.get_neighbors(self.in_pixels, x, y, self.height, self.width)
                if should_outline([self.in_pixels[(x, y)]] + neighbors):
                    self.out_pixels[(x, y)] = OUTLINE_COLOR

    def save(self):
        """Save out image."""
        self.out_image.save(self.out_name)
        pixels = self.final.load()
        for y in range(0, self.height-self.shrink_factor, self.shrink_factor):
            for x in range(0, self.width-self.shrink_factor, self.shrink_factor):
                pixels[(x//self.shrink_factor, y//self.shrink_factor)] = self.out_pixels[(x, y)]

        self.final.save("out_final.png")

def should_outline(to_check):
    """Check if pixel should be outlined based on its neighbors."""
    for pair in itertools.product(to_check, repeat=2):
        if max(list((abs(a-b) for a, b in zip(*pair)))) > OUTLINE_CUTOFF:
            return True

    return False

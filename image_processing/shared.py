"""Shared methods for outline and seamcarve."""
from os import path
from timeit import default_timer as timer

import numpy
from PIL import Image

HERE = path.abspath(path.dirname(__file__))

IN_NAME = path.join(HERE, "in.jpg")
OUT_NAME = path.join(HERE, "out.png")

BG_COLOR = (255, 255, 255, 254)

def get_neighbors(pixels, x, y, height, width, diagonals_on=False, flatten_and_filter=True):
    """Get neighbors of a pixel."""
    neighbors = [[(0, 0, 0, 0) for w in range(3)] for h in range(3)]
    # Left.
    if x > 0:
        neighbors[1][0] = pixels[(x-1, y)]
    # Right.
    if x < width-1:
        neighbors[1][2] = pixels[(x+1, y)]
    # Top.
    if y > 0:
        neighbors[0][1] = pixels[(x, y-1)]
    # Bottom.
    if y < height-1:
        neighbors[2][1] = pixels[(x, y+1)]

    # Diagonals.
    if diagonals_on:
        # Upper left.
        if x > 0 and y > 0:
            neighbors[0][0] = pixels[(x-1, y-1)]
        # Upper right
        if x < width-1 and y > 0:
            neighbors[0][2] = pixels[(x+1, y-1)]
        # Lower right.
        if x < width-1 and y < height-1:
            neighbors[2][2] = pixels[(x+1, y+1)]
        # Lower left.
        if x > 0 and y < height-1:
            neighbors[2][0] = pixels[(x-1, y+1)]

    if flatten_and_filter:
        return [item for sublist in neighbors for item in sublist if item is not None]

    return neighbors

def save_pixels(pixels):
    """Save array of pixels to an RGBA image."""
    print("Saving pixels to out image...")
    start = timer()
    array = numpy.array(pixels)
    out_image = Image.fromarray(array)
    out_image.save("seamcarved_image.png")
    end = timer()
    print(f"Took {end-start} seconds.")

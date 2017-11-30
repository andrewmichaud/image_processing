"""Methods for simple seam carving."""
import collections
import math
import sys

import numpy
from PIL import Image
from scipy.ndimage import filters

import shared

class SeamCarveData:
    """Data for seamcarving."""
    def __init__(self, x=0, y=0):
        self.energy = 0
        self.x = x
        self.y = y
        self.rel_x = x
        self.parent_choices = []
        self.parent = None
        self.children = []

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (self.x == other.x and self.y == other.y)
        else:
            return False

    def fix_rel_x(self):
        """Fix rel x."""
        self.rel_x -= 1
        return self

    def choose_parent(self):
        """Choose parent and update my energy."""
        if self.parent is None:
            old_parent_energy = 0
        else:
            old_parent_energy = self.parent.energy

        new_parent_energy = self.parent_choices[0].energy
        self.parent = self.parent_choices[0]
        for choice in self.parent_choices:
            if choice.energy < new_parent_energy:
                self.parent = choice
                new_parent_energy = choice.energy

        self.energy = self.energy - old_parent_energy + new_parent_energy

    def choose_children(self, energies):
        """Pick children for updating later if we get chosen as part of a seam."""

        # No children in the last row.
        if self.y == len(energies) - 1:
            return

        # Parents only need to choose children once.
        if len(self.children) > 0:
            return

        # Ensure we always have three children, to make later logic simpler.
        # Left.
        if self.x > 0:
            self.children.append(energies[self.y+1][self.x-1])
        else:
            self.children.append(None)

        # Middle
        self.children.append(energies[self.y+1][self.x])

        # Right.
        if self.x < len(energies[0]) - 1:
            self.children.append(energies[self.y+1][self.x+1])
        else:
            self.children.append(None)

def vertical_seamcarve(in_name=shared.IN_NAME, percent=90, show_carve=True, show_energy=True):
    """Seamcarve image N times."""
    image = Image.open(in_name).convert("L")
    im_pixels = image.load()

    grad_mags = get_gradient_magnitudes(image)
    # mag_im = Image.fromarray(grad_mags).convert("RGBA")
    # mag_im_pixels = mag_im.load()

    (energies, (min_e, max_e)) = calculate_sc_datas(image, grad_mags)

    # Create copy of out pixels we can manipulate and then save.
    out_pixels = []
    for y in range(image.height):
        row = []
        for x in range(image.width):
            if show_energy:
                # Compress energy into something that fits in 0-255.
                energy = energies[y][x].energy
                comp = int(numpy.interp(energy, [min_e, max_e], [0, 255]))
                row.append((comp, comp, comp, 254))
            else:
                row.append(im_pixels[(x, y)])

        out_pixels.append(row)


    # Carve!
    count = math.floor(min(((100-percent) / 100) * image.width, image.width))
    for i in range(count):

        # Get sorted list of bottom row to get our seams.
        seam_starts = sorted(energies[-1], key=lambda sc_data: sc_data.energy)
        min_sc_data = seam_starts[i]

        # Get top of seam.
        seam = collections.deque([min_sc_data])
        parent = min_sc_data.parent
        while parent is not None:
            seam.appendleft(parent)
            parent = parent.parent

        # Iterate over seam twice, once to draw/remove seam and once to update energies.
        for elem in seam:

            # Update out for each pixel in seam.
            if show_carve:
                out_pixels[elem.y][elem.x] = (255, 0, 0, 254)
            else:
                out_pixels[elem.y] = out_pixels[elem.y][:elem.rel_x] + \
                        out_pixels[elem.y][elem.rel_x+1:]

            # Remove pixel from energies.
            energies[elem.y] = energies[elem.y][:elem.rel_x] +\
                    list(map(lambda x: x.fix_rel_x(), energies[elem.y][elem.rel_x+1:]))

        for s, elem in enumerate(seam):
            # We need to update some energies, but we don't want to have to update the entire
            # image.
            # That's slow, and it's a waste. Not every pixel is affected by this seam being
            # removed.
            # This diagram shows which pixels (spoilers - a cone).
            # x pixels need cost updated and need parents re-chosen.
            #_ pixels just the cost update.
            # At this point, we've removed all 'o's, this is just to understand the whole picture.
            #            o
            #           xox
            #          _oxx_
            #         _oxx___
            #        _oxx_____
            #       _oxx_______
            #      _oxx_________
            #     _xxo___________
            #    ___xxo___________
            #   _____oxx___________
            #  _____oxx_____________
            # _____oxx_______________

            # Skip first row, they don't need to be updated.
            if s == 0:
                continue

            # Update pixels affected by this pixel deletion.
            # Slicing saves us some effort by automatically handling going past the end of the
            # list.
            # Have to handle going past the start ourselves.
            # We want to get the s pixels on either side of the one we deleted, keeping in mind we
            # moved everything to the right over by one already.
            start = elem.rel_x - s
            if start < 0:
                start = 0
            end = elem.rel_x + s

            affected = energies[s][start:end]

            for sc_data in affected:

                # Re-choose parent options, minding edges.
                sc_data.parent_choices = []
                # Left.
                if sc_data.rel_x > 0:
                    sc_data.parent_choices.append(energies[sc_data.y-1][sc_data.rel_x-1])

                # Middle.
                sc_data.parent_choices.append(energies[sc_data.y-1][sc_data.rel_x])

                # Right.
                if sc_data.rel_x < len(energies[sc_data.y-1])-1:
                    sc_data.parent_choices.append(energies[sc_data.y-1][sc_data.rel_x+1])

                # And make sure to update energies!
                sc_data.choose_parent()
                energies[sc_data.y][sc_data.rel_x] = sc_data

    return out_pixels

def get_gradient_magnitudes(image):
    """Get gradient magnitude array."""
    array = numpy.asarray(image)

    imx = numpy.zeros(array.shape)
    filters.sobel(array, 1, imx)

    imy = numpy.zeros(array.shape)
    filters.sobel(array, 0, imy)

    magnitude = numpy.sqrt(imx**2 + imy**2)
    return magnitude

def calculate_sc_datas(image, grad_mags):
    """Calculate grid of energies for image."""
    width = image.width
    height = image.height

    min_e = sys.maxsize
    max_e = 0

    energies = []
    for y in range(height):

        row = []
        for x in range(width):

            # Create item, get energy.
            sc_data = SeamCarveData(x=x, y=y)
            sc_data.energy = grad_mags[(y, x)]

            # Update parents.
            if y > 0:
                # Choose parent options, minding edges.
                # Left.
                if x > 0:
                    sc_data.parent_choices.append(energies[-1][x-1])

                # Middle.
                sc_data.parent_choices.append(energies[-1][x])

                # Right.
                if x < width - 1:
                    sc_data.parent_choices.append(energies[-1][x+1])

                # Choose cheapest parent.
                sc_data.choose_parent()

            # Set parent for element two rows ago, who has a whole row below it to work with.
            if y > 1:
                energies[-2][x].choose_children(energies)

            # Update min, max.
            if sc_data.energy > max_e:
                max_e = sc_data.energy

            if sc_data.energy < min_e:
                min_e = sc_data.energy

            row.append(sc_data)

        energies.append(row)

    # Handle choosing children.
    for y in range(height):
        for x in range(width):
            energies[y][x].choose_children(energies)

    return (energies, (min_e, max_e))

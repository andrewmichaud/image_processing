"""Methods for simple seam carving."""
import bisect
import collections
import math
from timeit import default_timer as timer
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
        self.parent_choices = []
        self.parent = None

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (self.x == other.x and self.y == other.y)

        return False

    def __lt__(self, other):
        if isinstance(self, other.__class__):
            return self.x < other.x

        raise TypeError

    def __gt__(self, other):
        if isinstance(self, other.__class__):
            return self.x > other.x

        raise TypeError

    def __le__(self, other):
        if isinstance(self, other.__class__):
            return self.x <= other.x

        raise TypeError

    def __ge__(self, other):
        if isinstance(self, other.__class__):
            return self.x >= other.x

        raise TypeError

    def __gt__(self, other):
        if isinstance(self, other.__class__):
            return self.x > other.x

        raise TypeError

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

def vertical_seamcarve(in_name=shared.IN_NAME, percent=90, show_carve=False, show_energy=False):
    """Seamcarve image N times."""
    print("Getting image open and making arrays...")
    image = Image.open(in_name).convert("RGBA")
    color_image = image.copy()
    lum_image = image.copy().convert("L")
    lum_array = numpy.asarray(lum_image)

    total_time = 0

    start = timer()
    print("Calculating grad mags...")
    grad_mags = get_gradient_magnitudes(lum_array)
    end = timer()
    total_time += end-start
    print(f"Took {end-start} seconds ({total_time} cumulative).")

    start = timer()
    print("Calculating sc_data...")
    (energies, (min_e, max_e)) = calculate_sc_datas(grad_mags)
    end = timer()
    total_time += end-start
    print(f"Took {end-start} seconds ({total_time} cumulative).")

    # Create copy of out pixels we can manipulate and then save.
    start = timer()
    print("Copying pixels to out array ...")
    # TODO slow?
    if show_energy:
        def lum2rgba(energy):
            return (energy, energy, energy, 254)

        out_pixels = numpy.array([[lum2rgba(sc_data.energy) for sc_data in energy_row]
                                  for energy_row in energies])

    else:
        out_pixels = numpy.asarray(color_image).copy().reshape(image.height, image.width, 4)

    end = timer()
    total_time += end-start
    print(f"Took {end-start} seconds ({total_time} cumulative).")

    # Carve!
    start_all_carves = timer()
    count = math.floor(min(((100-percent) / 100) * image.width, image.width))
    print(f"Starting carving ({count} carves)...")
    for i in range(count):
        start_time = timer()
        print(f"Starting carve {i+1}/{count}...")

        # Get sorted list of bottom row to get our seams.
        seam_starts = sorted(energies[-1], key=lambda sc_data: sc_data.energy)
        min_sc_data = seam_starts[i]

        # Get top of seam.
        seam = collections.deque([min_sc_data])
        parent = min_sc_data.parent
        while parent is not None:
            seam.appendleft(parent)
            parent = parent.parent

        # Append dummy elements, so we can enumerate a bit farther and handle outputting,
        # deleting elements, and updating elements, all in the same loop.
        seam.append(None)

        for sindex, elem in enumerate(seam):
            # Don't try to update the dummy element.
            if elem is not None:
                # Get index of elem in its row.
                elem_row_index = bisect.bisect_left(energies[elem.y], elem)

                # Update out for each pixel in seam.
                if show_carve:
                    out_pixels[(elem.y, elem.x)] = (255, 0, 0, 254)
                else:
                    after = numpy.delete(out_pixels[elem.y], (elem_row_index), axis=0)
                    padded = numpy.concatenate((after, [[0, 0, 0, 0]]), axis=0)
                    out_pixels[elem.y] = padded

                # Remove pixel from energies.
                    del energies[elem.y][elem_row_index]

            # We need to update some energies, but we don't want to have to update the entire
            # image.
            # That's slow, and it's a waste. Not every pixel is affected by this seam being
            # removed.
            # This diagram shows which pixels (spoilers - a cone).
            # o pixels are the seam pixels
            # _ pixels might need updates
            # At this point, we've removed all 'o's, this is just to understand the whole picture.
            #            o
            #           _o_
            #          _o___
            #         _o_____
            #        _o_______
            #       _o_________
            #      _o___________
            #     ___o___________
            #    _____o___________
            #   _____o_____________
            #  _____o_______________
            # _____o_________________

            #
            #           __
            #          ____
            #         ______
            #        ________
            #       __________
            #      ____________
            #     ______________
            #    ________________
            #   __________________
            #  ____________________
            # ______________________

            # Process elements two rows ago, so we know they've been updated and their parents have
            # been updated.
            # Don't start updating too early.
            if sindex < 2:
                continue

            update_index = sindex - 1
            update_elem = seam[update_index]
            # Get index of update_elem in its row.
            update_elem_row_index = bisect.bisect_left(energies[update_index], update_elem)

            # Update pixels affected by this pixel deletion.
            # Slicing saves us some effort by automatically handling going past the end of the
            # list.
            # Have to handle going past the start ourselves.
            # We want to get the s pixels on either side of the one we deleted, keeping in mind we
            # moved everything to the right over by one already.
            start = max(update_elem_row_index - update_index, 0)
            end = update_elem_row_index + update_index
            affected = energies[update_index][start:end]

            for sc_data in affected:
                # Get index of affected in the row.
                affected_index = bisect.bisect_left(energies[update_index], sc_data)

                # Re-choose parent options, minding edges.
                sc_data.parent_choices = []
                # Left.
                if affected_index > 0:
                    sc_data.parent_choices.append(energies[update_index-1][affected_index-1])

                # Middle.
                sc_data.parent_choices.append(energies[update_index-1][affected_index])

                # Right.
                if affected_index < len(energies[update_index]) - 1:
                    sc_data.parent_choices.append(energies[update_index-1][affected_index+1])

                # And make sure to update energies!
                sc_data.choose_parent()
                energies[update_index][affected_index] = sc_data

        end_time = timer()
        total_time += end_time-start_time
        print(f"Took {end_time-start_time} seconds ({total_time} cumulative).")

    out_pixels = numpy.hsplit(out_pixels, [image.width-count, image.width-count])[0]

    end_all_carves = timer()
    print(f"Took {end_all_carves-start_all_carves} seconds ({total_time} cumulative).")

    return out_pixels

def get_gradient_magnitudes(array):
    """Get gradient magnitude array."""
    imx = numpy.zeros(array.shape)
    filters.sobel(array, 1, imx)

    imy = numpy.zeros(array.shape)
    filters.sobel(array, 0, imy)

    return numpy.sqrt(imx**2 + imy**2)

def calculate_sc_datas(grad_mags):
    """Calculate grid of energies for image."""
    (height, width) = numpy.shape(grad_mags)

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

            # Update min, max.
            if sc_data.energy > max_e:
                max_e = sc_data.energy

            if sc_data.energy < min_e:
                min_e = sc_data.energy

            row.append(sc_data)

        energies.append(row)

    return (energies, (min_e, max_e))

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
""" Functions to generate terrain. """

import numpy as np
from scipy import ndimage

from improver.synthetic_data.diamond_square import diamond_square

MAX_HEIGHT = 1000
PERCENTILE = 25


def _set_initial_sea(terrain):
    """ If 1 to 3 corners are negative, set two rows as sea """
    x1 = y1 = x2 = y2 = None
    indices = np.array([0, -1])
    corners = np.array([terrain[0, 0], terrain[0, -1], terrain[-1, 0], terrain[-1, -1]])

    neg_corners = corners[np.where(corners < 0)[0]]

    if len(neg_corners) > 0 and len(neg_corners) != 4:
        # Find smallest corner
        min_corner = np.amin(neg_corners)
        i = np.where(corners == min_corner)[0][0]

        # Get min corner indices
        x1 = y1 = None

        if i == 0 or i == 1:
            x1 = 0
        elif i == 2 or i == 3:
            x1 = -1

        if i == 0 or i == 2:
            y1 = 0
        elif i == 1 or i == 3:
            y1 = -1

        # Get index values to check other corner values
        x_check = indices[np.where(indices != x1)[0][0]]
        y_check = indices[np.where(indices != y1)[0][0]]

        if terrain[x1, y_check] < 0 and terrain[x_check, y1] < 0:
            # If both adjacent corners are < 0, set x2 y2 as the indices of the corner with the smaller value
            x2, y2 = (
                (x1, y_check)
                if terrain[x1, y_check] < terrain[x_check, y1]
                else (x_check, y1)
            )
        elif terrain[x1, y_check] < 0:
            # If the adjacent corner at x,y_check is < 0, set x2,y2 as this corner
            x2 = x1
            y2 = y_check
        elif terrain[x_check, y1] < 0:
            # If the adjacent corner at x_check,y is < 0, set x2,y2 as this corner
            x2 = x_check
            y2 = y1
        elif terrain[x1, y_check] < terrain[x_check, y1]:
            # If neither of the adjacent corners are < 0 but the value at x,y_check is lower, set x2,y2 as that corner
            x2 = x1
            y2 = y_check
        elif terrain[x_check, y1] < terrain[x1, y_check]:
            # If neither of the adjacent corners are < 0 but the value at x_check,y is lower, set x2,y2 as that corner
            x2 = x_check
            y2 = y1

        if None not in [x1, y1, x2, y2]:
            if x1 == x2 == 0:
                terrain[0:2, :] = -0.01
            elif x1 == x2 == -1:
                terrain[-2:, :] = -0.01
            elif y1 == y2 == 0:
                terrain[:, 0:2] = -0.01
            elif y1 == y2 == -1:
                terrain[:, -2:] = -0.01

    return terrain


def _extract_domain(terrain, npoints):
    """ If the terrain array generated is bigger than the domain size requested,
    extract a subset of the requested size """
    if npoints < len(terrain):
        x_lower_bound = x_upper_bound = y_lower_bound = y_upper_bound = None

        # Find the indices of the maximum value
        max_i = np.where(terrain == np.amax(terrain))

        neg_indices = np.where(terrain < 0)

        if len(np.unique(neg_indices[0])) > len(np.unique(neg_indices[1])):
            # If there are more negative values in x-direction, y bounds need to be set
            # to capture this, set x bounds around maximum value
            x_lower_bound = int(max_i[0] - npoints) if max_i[0] >= npoints else 0
            x_upper_bound = x_lower_bound + npoints

            if len(np.where(terrain[:, 0] < 0)[0]) > len(
                np.where(terrain[:, -1] < 0)[0]
            ):
                # If there are more negative values at y == 0, set y bounds towards y == 0
                boundary_i = None
                for x in range(x_lower_bound, x_upper_bound):
                    max_row = np.amax(np.where(terrain[x, :] < 0))
                    if boundary_i is None or max_row < boundary_i:
                        boundary_i = max_row

                twenty_pcnt = int(np.ceil(npoints * 0.2))

                y_lower_bound = (
                    int(boundary_i - twenty_pcnt) if twenty_pcnt <= boundary_i else 0
                )
                y_upper_bound = y_lower_bound + npoints
                if y_upper_bound > len(terrain):
                    diff = y_upper_bound - (len(terrain) - 1)
                    y_upper_bound -= diff
                    y_lower_bound -= diff
            elif len(np.where(terrain[:, 0] < 0)[0]) < len(
                np.where(terrain[:, -1] < 0)[0]
            ):
                # If there are more negative values at y == -1, set y bounds towards y == -1
                boundary_i = None
                for x in range(x_lower_bound, x_upper_bound):
                    max_row = np.amin(np.where(terrain[x, :] < 0))
                    if boundary_i is None or max_row < boundary_i:
                        boundary_i = max_row

                twenty_pcnt = int(np.ceil(npoints * 0.2))

                y_upper_bound = (
                    int(boundary_i + twenty_pcnt)
                    if boundary_i + twenty_pcnt < len(terrain)
                    else len(terrain) - 1
                )
                y_lower_bound = y_upper_bound - npoints
                if y_lower_bound < 0:
                    diff = y_lower_bound
                    y_upper_bound -= diff
                    y_lower_bound -= diff
        elif len(np.unique(neg_indices[0])) < len(np.unique(neg_indices[1])):
            # If there are more negative values in y-direction, x bounds need to be set
            # to capture this, set y bounds around maximum value
            y_lower_bound = int(max_i[1] - npoints) if max_i[1] >= npoints else 0
            y_upper_bound = y_lower_bound + npoints

            if len(np.where(terrain[0, :] < 0)[0]) > len(
                np.where(terrain[-1, :] < 0)[0]
            ):
                # If there are more negative values at x == 0, set x bounds towards x == 0
                boundary_i = None
                for y in range(y_lower_bound, y_upper_bound):
                    max_row = np.amax(np.where(terrain[:, y] < 0))
                    if boundary_i is None or max_row < boundary_i:
                        boundary_i = max_row

                twenty_pcnt = int(np.ceil(npoints * 0.2))

                x_lower_bound = (
                    int(boundary_i - twenty_pcnt) if twenty_pcnt <= boundary_i else 0
                )
                x_upper_bound = x_lower_bound + npoints
                if x_upper_bound > len(terrain):
                    diff = x_upper_bound - (len(terrain) - 1)
                    x_upper_bound -= diff
                    x_lower_bound -= diff

            elif len(np.where(terrain[0, :] < 0)[0]) < len(
                np.where(terrain[-1, :] < 0)[0]
            ):
                # If there are more negative values at x == -1, set x bounds towards x == -1
                boundary_i = None
                for y in range(y_lower_bound, y_upper_bound):
                    max_row = np.amin(np.where(terrain[:, y] < 0))
                    if boundary_i is None or max_row < boundary_i:
                        boundary_i = max_row

                twenty_pcnt = int(np.ceil(npoints * 0.2))

                x_upper_bound = (
                    int(boundary_i + twenty_pcnt)
                    if boundary_i + twenty_pcnt < len(terrain)
                    else len(terrain) - 1
                )
                x_lower_bound = x_upper_bound - npoints
                if x_lower_bound < 0:
                    diff = x_lower_bound
                    x_upper_bound -= diff
                    x_lower_bound -= diff
        else:
            # Extract if no negatives
            pass

        if None not in [x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound]:
            terrain = terrain[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound]

    return terrain


def _adjust_values(terrain):
    """ Modify the values generated to adjust the terrain shape and scale to suitable
    height values """
    max_val = np.amax(terrain)
    terrain = terrain * (1.0 / max_val)

    terrain = np.power(terrain, 3)

    terrain = terrain * MAX_HEIGHT

    return terrain


def _smooth_data(terrain, spatial_grid, grid_spacing):
    """ Apply Gaussian filter in order to reduce the "jaggedness" of the generated values
    and also ensure that this is maintained regardless of grid spacing """
    # Set sigma to 0.8 for grid_spacing of 0.02/2000m
    sigma = 0.8

    if spatial_grid == "equalarea" and grid_spacing != 2000:
        sigma *= 2000 / grid_spacing
    elif spatial_grid == "latlon" and grid_spacing != 0.02:
        sigma *= 0.02 / grid_spacing

    filtered_data = ndimage.gaussian_filter(terrain, sigma)

    return filtered_data


def _adjust_land_sea_ratio(filtered_data):
    """ Adjust the ratio of land to sea in the generated terrain by subtracting the
    value of the specified percentile if the percentage of sea is less than this
    amount """
    # Check percentage of sea
    sea_percentage = (len(filtered_data[filtered_data < 0]) / filtered_data.size) * 100

    modified_terrain = np.copy(filtered_data)

    # If percentage of sea is < PERCENTILE%, get the PERCENTILEth percentile and shift all values down
    if sea_percentage < PERCENTILE:
        percentile = np.percentile(filtered_data, PERCENTILE)

        modified_terrain -= percentile

        # Convert to binary and fill holes
        binary = np.copy(modified_terrain)
        binary = np.where(binary > 0, 1, 0)

        # Copy binary array
        filled_holes = np.copy(binary)

        # Set "land" edge and sides (up to the last boundary of land and sea) from this
        # edge to 1 to ensure holes at edge removed
        if filtered_data[0, 0] > 0 and filtered_data[0, -1] > 0:
            filled_holes[0, :] = 1
            filled_holes[0 : np.amax(np.where(binary[:, 0] > 0)), 0] = 1
            filled_holes[0 : np.amax(np.where(binary[:, -1] > 0)) :, -1] = 1
        elif filtered_data[0, 0] > 0 and filtered_data[-1, 0] > 0:
            filled_holes[:, 0] = 1
            filled_holes[0, 0 : np.amax(np.where(binary[:, 0] > 0))] = 1
            filled_holes[-1, 0 : np.amax(np.where(binary[:, -1] > 0))] = 1
        elif filtered_data[-1, -1] > 0 and filtered_data[0, -1] > 0:
            filled_holes[:, -1] = 1
            filled_holes[0, np.amin(np.where(binary[:, 0] > 0)) :] = 1
            filled_holes[-1, np.amin(np.where(binary[:, -1] > 0)) :] = 1
        elif filtered_data[-1, -1] > 0 and filtered_data[-1, 0] > 0:
            filled_holes[-1, :] = 1
            filled_holes[np.amin(np.where(binary[:, 0] > 0)) :, 0] = 1
            filled_holes[np.amin(np.where(binary[:, -1] > 0)) :, -1] = 1

        # Fill holes
        filled_holes = ndimage.binary_fill_holes(filled_holes).astype(int)

        # Return any edges that were not holes to 0
        filled_holes[np.where((filled_holes[:, 0:2] == [1, 0]).all(axis=1)), 0] = 0
        filled_holes[np.where((filled_holes[:, -2:] == [0, 1]).all(axis=1)), -1] = 0
        filled_holes[np.where((filled_holes[0:2, :] == [[1], [0]]).all(axis=0)), 0] = 0
        filled_holes[np.where((filled_holes[-2:, :] == [[0], [1]]).all(axis=0)), -1] = 0

        # Determine where holes have been created when subtracting 25th percentile by comparing to original binary array
        holes = np.where(np.not_equal(binary, filled_holes))

        # Set values back to original values
        modified_terrain[holes] = filtered_data[holes]

    return modified_terrain


def _reduce_noise(terrain):
    """ Remove small "land" areas from the "sea" area """
    terrain_copy = np.copy(terrain)

    # Convert to binary
    binary = np.copy(terrain_copy)
    binary = np.where(binary > 0, 1, 0)

    # Remove noise
    noise_removed = np.copy(binary)
    binary_eroded = ndimage.binary_erosion(noise_removed, structure=np.ones((15, 15)))
    noise_removed = ndimage.binary_propagation(binary_eroded, mask=noise_removed)

    # Apply noise removal to terrain array and check is equal to noise removed binary
    terrain_copy[noise_removed == 0] = -0.01

    return noise_removed


def generate_terrain(npoints, spatial_grid, grid_spacing):
    """ Generate terrain by performing the diamond-square algorithm, and then adjusting
    the values in a series of additional steps.
    
    Args:
        npoints (int):
            Number of points along each of the y and x spatial axes.
        spatial_grid (str):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        grid_spacing (float):
            Resolution of grid (metres or degrees).

    Returns:
        np.ndarray:
            Array containing surface altitude data.
    """
    # Generate data with diamond-square algorithm
    terrain = diamond_square(npoints)

    # Set first two rows of data as sea
    terrain = _set_initial_sea(terrain)

    # Extract domain
    terrain_domain_extracted = _extract_domain(terrain, npoints)

    # Smooth data
    terrain_filtered = _smooth_data(
        terrain_domain_extracted, spatial_grid, grid_spacing
    )

    # Adjust values
    terrain_values_adjusted = _adjust_values(terrain_filtered)

    # Adjust land sea ratio
    terrain_land_sea_modified = _adjust_land_sea_ratio(terrain_values_adjusted)

    # Remove noise
    terrain_noise_removed = _reduce_noise(terrain_land_sea_modified)

    return terrain_noise_removed

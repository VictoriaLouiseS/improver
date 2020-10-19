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
"""Module containing an implementation of the diamond-square algorithm for terrain generation."""

import math
import random

import numpy as np

INIT_VAL = 5555.5
CORNER_VALUES = [-0.6, -0.8, 0.2, 1.0]


def _initialise_terrain(npoints):
    """ Create an array filled with INIT_VAL (array must be of width/height 2^n + 1
    for diamond-square algorithm) """
    # Create array of INIT_VAL to fill (array must be of width/height 2^n + 1 for diamond-square algorithm)
    n = math.ceil(math.log2(npoints - 1))
    terrain_length = int(math.pow(2, n)) + 1
    terrain = np.full([terrain_length, terrain_length], INIT_VAL)

    # Initialise corner values, set two or three corners -ve to create coastal region
    terrain[0, 0] = CORNER_VALUES[0]
    terrain[0, -1] = CORNER_VALUES[1]
    terrain[-1, 0] = CORNER_VALUES[2]
    terrain[-1, -1] = CORNER_VALUES[3]

    return terrain, n


def _get_midpoint_indices(i_diff, x, y):
    """ Calculate the indices of the centre of the square and the corner points. """
    # Calculate the indices of the centre point of the square
    midpoint_x = int(i_diff * ((2 * x) + 1))
    midpoint_y = int(i_diff * ((2 * y) + 1))

    # Calculate indices of corner limits
    x_lim_low = int(midpoint_x - i_diff)
    x_lim_high = int(midpoint_x + i_diff)
    y_lim_low = int(midpoint_y - i_diff)
    y_lim_high = int(midpoint_y + i_diff)

    return midpoint_x, midpoint_y, x_lim_low, x_lim_high, y_lim_low, y_lim_high


def _get_diamond_indices(x, y, i_diff, terrain_length):
    """ Calculate the indices to calculate the values for """
    x_lower = int(x - i_diff) if x >= i_diff else int(x - (i_diff + 1))
    x_higher = int(i_diff) if (x + i_diff) >= terrain_length else int(x + i_diff)
    y_lower = int(y - i_diff) if y >= i_diff else int(y - (i_diff + 1))
    y_higher = int(i_diff) if (y + i_diff) >= terrain_length else int(y + i_diff)

    return x_lower, x_higher, y_lower, y_higher


def _calculate_value(corner_vals, random_num_lim):
    """ Calculate the value to set in the array. """
    # Calculate average
    corner_avg = sum(corner_vals) / 4

    # Generate random number
    random_val = random.uniform(-random_num_lim, random_num_lim)

    return corner_avg + random_val


def _diamond_square_steps(terrain, step, npoints, max_random):
    """ Perform each diamond and square step for the iteration. """
    # Calculate the number of points to calculate the value for during the square step
    n_sq = int(math.pow(2, step))

    # Calculate the difference between indices
    i_diff = int(len(terrain) - 1) / (n_sq * 2)

    # Set the limits of the random number
    sq_random_num_lim = max_random / (step + 1)

    # Perform square step
    for sq_x in range(n_sq):
        for sq_y in range(n_sq):
            (
                midpoint_x,
                midpoint_y,
                x_lim_low,
                x_lim_high,
                y_lim_low,
                y_lim_high,
            ) = _get_midpoint_indices(i_diff, sq_x, sq_y)

            corner_vals = [
                terrain[x_lim_low, y_lim_low],
                terrain[x_lim_low, y_lim_high],
                terrain[x_lim_high, y_lim_low],
                terrain[x_lim_high, y_lim_high],
            ]

            # Assign value
            terrain[midpoint_x, midpoint_y] = _calculate_value(
                corner_vals, sq_random_num_lim
            )

    d_random_num_lim = max_random / (npoints - step)

    # Perform diamond step
    for sq_x in range(n_sq):
        for sq_y in range(n_sq):
            (
                midpoint_x,
                midpoint_y,
                x_lim_low,
                x_lim_high,
                y_lim_low,
                y_lim_high,
            ) = _get_midpoint_indices(i_diff, sq_x, sq_y)

            x_indices = x_lim_low, x_lim_high
            y_indices = y_lim_low, y_lim_high

            # For each of the x and y limits, calculate the square edge values
            for x in x_indices:
                if terrain[x, midpoint_y] == INIT_VAL:
                    # Get indices for diamond point values
                    x_lower, x_higher, y_lower, y_higher = _get_diamond_indices(
                        x, midpoint_y, i_diff, len(terrain)
                    )

                    corner_vals = [
                        terrain[x, y_lower],
                        terrain[x, y_higher],
                        terrain[x_lower, midpoint_y],
                        terrain[x_higher, midpoint_y],
                    ]

                    # Assign value
                    terrain[x, midpoint_y] = _calculate_value(
                        corner_vals, d_random_num_lim
                    )

            for y in y_indices:
                if terrain[midpoint_x, y] == INIT_VAL:
                    # Get indices for diamond points values
                    x_lower, x_higher, y_lower, y_higher = _get_diamond_indices(
                        midpoint_x, y, i_diff, len(terrain)
                    )

                    corner_vals = [
                        terrain[midpoint_x, y_lower],
                        terrain[midpoint_x, y_higher],
                        terrain[x_lower, y],
                        terrain[x_higher, y],
                    ]

                    # Assign value
                    terrain[midpoint_x, y] = _calculate_value(
                        corner_vals, d_random_num_lim
                    )

    return terrain


def diamond_square(npoints):
    """ Generate terrain using the diamond-square algorithm.
    
    Args:
        npoints (int):
            Number of points along each of the y and x spatial axes.

    Returns:
        np.ndarray:
            Array filled with values generated with the diamond-square algorithm.
     """
    # Initialise terrain array
    terrain, n = _initialise_terrain(npoints)

    # Perform diamond-square algorithm
    random.seed(0)

    max_random = 1

    for step in range(n):
        terrain = _diamond_square_steps(terrain, step, npoints, max_random)

    return terrain

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
""" Functions to create ancillary cubes containing synthetic data. """

import iris
import numpy as np

from improver.synthetic_data.generate_metadata import (
    DEFAULT_GRID_SPACING,
    DEFAULT_SPATIAL_GRID,
    generate_metadata,
)
from improver.synthetic_data.generate_terrain import generate_terrain

DEFAULT_SIZE = 71


def _create_orography_data(**kwargs):
    """ Create orography cube """
    # Get terrain data
    terrain_data = generate_terrain(
        kwargs["npoints"], kwargs["spatial_grid"], kwargs["grid_spacing"]
    )

    # Create metadata cube
    kwargs["name"] = "surface_altitude"
    kwargs["ensemble_members"] = 0
    orography_cube = generate_metadata(**kwargs)

    # Replace empty data with terrain data
    orography_cube.data = terrain_data

    return orography_cube


def _create_land_sea_mask_data(terrain, **kwargs):
    """ Create land/sea mask cube from orography data """
    # Create land sea mask data
    land_sea_mask_data = np.where(terrain <= 0, 0, 1)

    # Create metadata cube
    kwargs["name"] = "land_binary_mask"
    kwargs["ensemble_members"] = 0
    land_sea_mask_cube = generate_metadata(**kwargs)

    # Replace empty array with data
    land_sea_mask_cube.data = land_sea_mask_data

    return land_sea_mask_cube


def get_ancillaries(**kwargs):
    """ Generate an orography cube with surface altitude data and a land/sea mask cube
    with land binary mask data.

    Returns:
        Tuple(iris.cube.Cube, iris.cube.Cube):
            Cube containing surface altitude data and cube containing land binary mask
            data.
    """
    if "npoints" not in kwargs or kwargs["npoints"] is None:
        kwargs["npoints"] = DEFAULT_SIZE

    if "spatial_grid" not in kwargs or kwargs["spatial_grid"] is None:
        kwargs["spatial_grid"] = DEFAULT_SPATIAL_GRID

    if "grid_spacing" not in kwargs or kwargs["grid_spacing"] is None:
        kwargs["grid_spacing"] = DEFAULT_GRID_SPACING[kwargs["spatial_grid"]]

    # Create orography cube
    orography_cube = _create_orography_data(**kwargs)

    # Create land/sea mask cube
    land_sea_mask_cube = _create_land_sea_mask_data(orography_cube.data, **kwargs)

    return orography_cube, land_sea_mask_cube

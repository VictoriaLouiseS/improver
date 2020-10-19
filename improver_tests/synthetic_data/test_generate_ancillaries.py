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
"""Tests for ancillary cube generation."""

import iris
import numpy as np
import pytest

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.synthetic_data.diamond_square import INIT_VAL
from improver.synthetic_data.generate_ancillaries import DEFAULT_SIZE, get_ancillaries
from improver.synthetic_data.generate_metadata import DEFAULT_GRID_SPACING

SPATIAL_GRID_ATTRIBUTE_DEFAULTS = {
    "latlon": {
        "y": "latitude",
        "x": "longitude",
        "grid_spacing": 0.02,
        "units": "degrees",
        "coord_system": GLOBAL_GRID_CCRS,
    },
    "equalarea": {
        "y": "projection_y_coordinate",
        "x": "projection_x_coordinate",
        "grid_spacing": 2000,
        "units": "metres",
        "coord_system": STANDARD_GRID_CCRS,
    },
}


def test_generate_ancillaries_default():
    """ Test orography cube and land sea mask cube created with default settings. """
    orography_cube, land_sea_mask_cube = get_ancillaries()

    assert orography_cube.name() == "surface_altitude"
    assert orography_cube.units == "m"
    assert orography_cube.data.shape == (DEFAULT_SIZE, DEFAULT_SIZE)
    assert INIT_VAL not in orography_cube.data

    assert land_sea_mask_cube.name() == "land_binary_mask"
    assert land_sea_mask_cube.units == "1"
    assert land_sea_mask_cube.data.shape == (DEFAULT_SIZE, DEFAULT_SIZE)
    assert np.array_equal(land_sea_mask_cube.data, land_sea_mask_cube.data.astype(bool))


@pytest.mark.parametrize("npoints", (None, 900))
@pytest.mark.parametrize(
    "spatial_grid,grid_spacing",
    [("equalarea", None), ("equalarea", 1000), ("latlon", 0.01), ("latlon", None)],
)
def test_generate_ancillaries(npoints, spatial_grid, grid_spacing):
    """ Test orography cube and land sea mask cube created with different npoints,
    spatial grid and grid spacing """
    orography_cube, land_sea_mask_cube = get_ancillaries(
        npoints=npoints, spatial_grid=spatial_grid, grid_spacing=grid_spacing
    )

    if npoints is None:
        npoints = DEFAULT_SIZE

    if grid_spacing is None:
        grid_spacing = DEFAULT_GRID_SPACING[spatial_grid]

    spatial_grid_values = SPATIAL_GRID_ATTRIBUTE_DEFAULTS[spatial_grid]

    assert orography_cube.name() == "surface_altitude"
    assert orography_cube.units == "m"

    assert land_sea_mask_cube.name() == "land_binary_mask"
    assert land_sea_mask_cube.units == "1"

    for cube in (orography_cube, land_sea_mask_cube):
        assert cube.data.shape == (npoints, npoints)
        assert cube.coords()[0].name() == spatial_grid_values["y"]
        assert cube.coords()[1].name() == spatial_grid_values["x"]

        for axis in ("y", "x"):
            assert cube.coord(axis=axis).units == spatial_grid_values["units"]
            assert (
                cube.coord(axis=axis).coord_system
                == spatial_grid_values["coord_system"]
            )
            assert np.diff(cube.coord(axis=axis).points)[0] == pytest.approx(
                grid_spacing, abs=1e-5
            )

    assert orography_cube.data.shape == (npoints, npoints)
    assert INIT_VAL not in orography_cube.data

    assert np.array_equal(land_sea_mask_cube.data, land_sea_mask_cube.data.astype(bool))

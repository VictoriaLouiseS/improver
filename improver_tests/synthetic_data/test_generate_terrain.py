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
"""Tests for metadata cube generation."""

import math

import iris
import numpy as np
import pytest

from improver.synthetic_data.generate_terrain import _extract_domain, _set_initial_sea


@pytest.mark.parametrize(
    "corners,expected_rows",
    [
        ([0.8, 0.5, 0.3, 0.2], [None, None]),
        ([-0.8, -0.5, 0.3, 0.2], [0, None]),
        ([0.8, 0.5, -0.3, -0.2], [-1, None]),
        ([-0.8, 0.5, -0.3, 0.2], [None, 0]),
        ([0.8, -0.5, 0.3, -0.2], [None, -1]),
        ([-0.8, -0.5, -0.3, 0.2], [0, None]),
        ([0.8, -0.5, 0.3, 0.2], [None, -1]),
        ([-0.8, -0.5, -0.3, -0.2], [None, None]),
    ],
)
def test_set_initial_sea(corners, expected_rows):
    """ Test additional rows of sea set according to corner values """
    terrain = np.full((10, 10), 200.0, dtype=float)
    terrain[0, 0] = corners[0]
    terrain[0, -1] = corners[1]
    terrain[-1, 0] = corners[2]
    terrain[-1, -1] = corners[3]

    modified_terrain = _set_initial_sea(np.copy(terrain))

    if expected_rows[0] is not None:
        if expected_rows[0] == 0:
            assert np.all(modified_terrain[0:2, :] == -0.01)
            modified_terrain[-1, 0] = 200.0
            modified_terrain[-1, -1] = 200.0
            assert np.all(np.delete(modified_terrain, [0, 1], axis=0) == 200.0)
        elif expected_rows[0] == -1:
            assert np.all(modified_terrain[-2:, :] == -0.01)
            modified_terrain[0, 0] = 200.0
            modified_terrain[0, -1] = 200.0
            assert np.all(np.delete(modified_terrain, [8, 9], axis=0) == 200.0)
    elif expected_rows[1] is not None:
        if expected_rows[1] == 0:
            assert np.all(modified_terrain[:, 0:2] == -0.01)
            modified_terrain[0, -1] = 200.0
            modified_terrain[-1, -1] = 200.0
            assert np.all(np.delete(modified_terrain, [0, 1], axis=1) == 200.0)
        elif expected_rows[1] == -1:
            assert np.all(modified_terrain[:, -2:] == -0.01)
            modified_terrain[0, 0] = 200.0
            modified_terrain[-1, 0] = 200.0
            assert np.all(np.delete(modified_terrain, [8, 9], axis=1) == 200.0)
    elif expected_rows[0] is None and expected_rows[1] is None:
        np.testing.assert_array_equal(modified_terrain, terrain)


@pytest.mark.parametrize(
    "npoints,terrain_size,neg_rows",
    [
        (15, 30, [0, None]),
        (15, 30, [-1, None]),
        (15, 30, [None, 0]),
        (15, 30, [None, -1]),
        pytest.param(15, 30, [None, None], marks=pytest.mark.xfail),
        (30, 30, [0, None]),
    ],
)
def test_extract_domain(npoints, terrain_size, neg_rows):
    """ Test that domain is extracted from the generated terrain correctly """
    terrain = np.full((terrain_size, terrain_size), 5.0, dtype=float)

    terrain[(math.ceil(npoints / 2) + 3), math.ceil(npoints / 2)] = 100.0

    if neg_rows[0] is not None and neg_rows[1] is None:
        terrain[neg_rows[0], :] = -0.01
    elif neg_rows[0] is None and neg_rows[1] is not None:
        terrain[:, neg_rows[1]] = -0.01

    extracted_domain = _extract_domain(np.copy(terrain), npoints)

    if npoints == terrain_size:
        np.testing.assert_array_equal(terrain, extracted_domain)
    else:
        assert terrain.shape != extracted_domain.shape
        assert extracted_domain.shape == (npoints, npoints)

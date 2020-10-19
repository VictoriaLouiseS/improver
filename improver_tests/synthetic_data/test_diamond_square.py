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

from improver.synthetic_data.diamond_square import (
    CORNER_VALUES,
    INIT_VAL,
    _initialise_terrain,
    diamond_square,
)


@pytest.mark.parametrize("npoints,expected_size", [(71, 129), (129, 129)])
def test_initialise_terrain(npoints, expected_size):
    """ Test terrain array initialised correctly """
    terrain, n = _initialise_terrain(npoints)

    assert terrain.shape == (expected_size, expected_size)
    assert n == math.log2(expected_size - 1)

    assert terrain[0, 0] == CORNER_VALUES[0]
    assert terrain[0, -1] == CORNER_VALUES[1]
    assert terrain[-1, 0] == CORNER_VALUES[2]
    assert terrain[-1, -1] == CORNER_VALUES[3]

    assert np.all(
        np.delete(
            terrain,
            [
                0,
                (expected_size - 1),
                math.pow(expected_size, 2) - expected_size,
                math.pow(expected_size, 2) - 1,
            ],
        )
        == INIT_VAL
    )


@pytest.mark.parametrize("npoints,expected_size", [(71, 129), (129, 129)])
def test_diamond_square(npoints, expected_size):
    """ Test array values set using diamond-square algorithm """
    terrain = diamond_square(npoints)

    assert terrain.shape == (expected_size, expected_size)

    assert terrain[0, 0] == CORNER_VALUES[0]
    assert terrain[0, -1] == CORNER_VALUES[1]
    assert terrain[-1, 0] == CORNER_VALUES[2]
    assert terrain[-1, -1] == CORNER_VALUES[3]

    assert np.all(terrain != INIT_VAL)

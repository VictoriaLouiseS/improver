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
"""Module to generate a metadata cube."""

from datetime import datetime, timedelta

import numpy as np
from iris.std_names import STD_NAMES
from iris.util import squeeze

from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)

DEFAULT_RESOLUTION = {"latlon": 0.02, "equalarea": 2000}


def _get_units(name):
    """ Get output variable units from iris.std_names.STD_NAMES """
    try:
        units = STD_NAMES[name]["canonical_units"]
    except KeyError:
        raise ValueError("Units of {} are not known.".format(name))

    return units


def _create_time_bounds(time, time_period):
    """ Create time bounds using time - time_period as the lower bound and time as the
    upper bound"""
    lower_bound = time - timedelta(minutes=time_period)
    upper_bound = time

    return (lower_bound, upper_bound)


def _create_data_array(ensemble_members, leading_dimension, npoints, height_levels):
    """ Create data array of specified shape filled with zeros """
    if leading_dimension is not None:
        nleading_dimension = len(leading_dimension)
    else:
        nleading_dimension = ensemble_members

    if height_levels is not None:
        nheight_levels = len(height_levels)
    else:
        nheight_levels = None

    data_shape = []

    if nleading_dimension > 1:
        data_shape.append(nleading_dimension)
    if nheight_levels is not None:
        data_shape.append(nheight_levels)

    data_shape.append(npoints)
    data_shape.append(npoints)

    return np.zeros(data_shape, dtype=np.float32)


def generate_metadata(
    name="air_pressure_at_sea_level",
    units=None,
    spatial_grid="latlon",
    time=datetime(2017, 11, 10, 4, 0),
    time_period=None,
    frt=datetime(2017, 11, 10, 0, 0),
    ensemble_members=8,
    leading_dimension=None,
    percentile=False,
    probability=False,
    spp__relative_to_threshold="above",
    attributes=None,
    resolution=None,
    domain_corner=None,
    npoints=71,
    height_levels=None,
    pressure=False,
):
    """ Generate a cube with metadata only.

    Args:
        name (str):
            Output variable name, or if creating a probability cube the name of the
            underlying variable to which the probability field applies.
        units (Optional[str]):
            Output variable units, or if creating a probability cube the units of the
            underlying variable / threshold.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[datetime.datetime]):
            Single cube validity time. If time period given, time is used as the upper
            time bound.
        time_period (Optional[int]):
            The period in minutes between the time bounds. This is used to calculate
            the lower time bound.
        frt (Optional[datetime.datetime]):
            Single cube forecast reference time.
        ensemble_members (Optional[int]):
            Number of ensemble members. Default 8, unless percentile or probability set
            to True.
        leading_dimension (Optional[List[float]]):
            List of realizations, percentiles or thresholds.
        percentile (Optional[bool]):
            Flag to indicate whether the leading dimension is percentile values. If
            True, a percentile cube is created.
        probability (Optional[bool]):
            Flag to indicate whether the leading dimension is threshold values. If
            True, a probability cube is created.
        spp__relative_to_threshold (Optional[str]):
            Value of the attribute "spp__relative_to_threshold" which is required for
            IMPROVER probability cubes.
        attributes (Optional[Dict]):
            Dictionary of additional metadata attributes.
        resolution (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for
            equalarea).
        npoints (Optional[int]):
            Number of points along each of the y and x spatial axes.
        height_levels (Optional[List[float]]):
            List of altitude/pressure levels.
        pressure (Optional[bool]):
            Flag to indicate whether the height levels are specified as pressure, in
            Pa. If False, use height in metres.

    Returns:
        iris.cube.Cube:
            Output of set_up_variable_cube(), set_up_percentile_cube() or
            set_up_probability_cube()
    """
    if spatial_grid not in ("latlon", "equalarea"):
        raise ValueError(
            "Spatial grid {} not supported. Choose either latlon or equalarea.".format(
                spatial_grid
            )
        )

    if percentile is True and probability is True:
        raise ValueError(
            "Only percentile cube or probability cube can be generated, not both."
        )

    if domain_corner is not None and len(domain_corner) != 2:
        raise ValueError("Domain corner must be a list or tuple of length 2.")

    if units is None:
        units = _get_units(name)

    # If time_period specified, create time bounds using time as upper bound
    if time_period is not None:
        time_bounds = _create_time_bounds(time, time_period)
    else:
        time_bounds = None

    # If resolution not specified, use default for requested spatial grid
    if resolution is None:
        resolution = DEFAULT_RESOLUTION[spatial_grid]

    # Create ndimensional array of zeros
    data = _create_data_array(
        ensemble_members, leading_dimension, npoints, height_levels
    )

    # Set args to pass to cube set up function
    common_args = {
        "spatial_grid": spatial_grid,
        "time": time,
        "time_bounds": time_bounds,
        "frt": frt,
        "attributes": attributes,
        "grid_spacing": resolution,
        "domain_corner": domain_corner,
        "height_levels": height_levels,
        "pressure": pressure,
    }

    # Set up requested cube
    if percentile:
        metadata_cube = set_up_percentile_cube(
            data, percentiles=leading_dimension, name=name, units=units, **common_args,
        )
    elif probability:
        metadata_cube = set_up_probability_cube(
            data,
            leading_dimension,
            variable_name=name,
            threshold_units=units,
            spp__relative_to_threshold=spp__relative_to_threshold,
            **common_args,
        )
    else:
        metadata_cube = set_up_variable_cube(
            data, name=name, units=units, realizations=leading_dimension, **common_args,
        )

    metadata_cube = squeeze(metadata_cube)

    return metadata_cube

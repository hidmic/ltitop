# -*- coding: utf-8 -*-

# ltitop - A toolkit to describe and optimize LTI systems topology
# Copyright (C) 2021 Michel Hidalgo <hid.michel@gmail.com>
#
# This file is part of ltitop.
#
# ltitop is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ltitop is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with ltitop.  If not, see <http://www.gnu.org/licenses/>.

import scipy.signal as signal
from numpy.testing import assert_almost_equal

from ltitop.arithmetic.interval import interval
from ltitop.models.analysis import (
    dc_gain,
    is_stable,
    output_range,
    spectral_radius,
    worst_case_peak_gain,
)


def test_model_stability():
    model = signal.dlti([1], [1, 1])
    assert not is_stable(model)
    assert_almost_equal(spectral_radius(model), 1)
    model = signal.dlti([1], [1, 0.5])
    assert is_stable(model)
    assert_almost_equal(spectral_radius(model), 0.5)


def test_model_dc_gain():
    assert dc_gain(signal.dlti([0.75, 0], [1, -0.5])) == 1.5


def test_model_worst_case_peak_gain():
    assert worst_case_peak_gain(signal.dlti([0.75, 0], [1, -0.5])) == 1.5
    assert worst_case_peak_gain(signal.dlti([0.75, -0.75], [1, 0.0])) == 1.5


def test_model_output_range():
    model = signal.dlti([0.75, 0], [1, -0.5])
    input_range = interval(-1, 1)
    assert output_range(model, input_range) == interval(-1.5, 1.5)

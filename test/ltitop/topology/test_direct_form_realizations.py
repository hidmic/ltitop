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

import itertools

import pytest
import scipy.signal as signal
from numpy.testing import assert_allclose

from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.rounding import nearest_integer
from ltitop.topology.realizations.direct_forms import (
    DirectFormI,
    DirectFormII,
    TransposedDirectFormII,
)


@pytest.mark.parametrize(
    "realization_type,model",
    itertools.product(
        [DirectFormI, DirectFormII, TransposedDirectFormII],
        [
            signal.dlti([1, 0], [1, -0.5]),
            signal.dlti([1], [1, 0.5]),
            signal.dlti([1], [1, 2, 1]),
            signal.dlti([1, 2, 1], [1, 0, 0]),
        ],
    ),
)
def test_models_are_realized(realization_type, model):
    block = realization_type.from_model(model)
    assert_allclose(model.freqresp()[1], block.model.freqresp()[1])

    _, outputs = block.process(inputs=[1.0] + [0.0] * 99)
    _, (impulse_response,) = model.impulse(n=100)
    assert_allclose(outputs, impulse_response)


@pytest.mark.parametrize(
    "realization_type", [DirectFormI, DirectFormII, TransposedDirectFormII]
)
def test_realization_error_bounds(realization_type):
    model = signal.dlti([1], [1, -0.5])  # y[n] = x[n - 1] - 0.5 y[n - 1]
    block = realization_type.from_model(model)
    with FixedFormatArithmeticLogicUnit(
        format_=Q(7),
        allows_overflow=False,
        rounding_method=nearest_integer,
    ):
        assert block.computation_error_bounds(
            interval(-0.25, 0.25)
        ) in 10 * nearest_integer.error_bounds(
            -7
        )  # reasonable tolerance

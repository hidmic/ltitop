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

import pytest
import sympy

from ltitop.algorithms import Algorithm
from ltitop.algorithms.analysis import implementation_hardware
from ltitop.algorithms.statements import Assignment


@pytest.fixture
def exponential_filter_algo():
    x = sympy.IndexedBase("x")
    y = sympy.IndexedBase("y")
    k, alpha = sympy.symbols("k α")

    return Algorithm(
        inputs=x[k],
        states=y[k - 1],
        outputs=y[k],
        parameters=alpha,
        procedure=[
            Assignment(lhs=y[k], rhs=alpha * x[k] + (1 - alpha) * y[k - 1]),
            Assignment(lhs=y[k - 1], rhs=y[k]),
        ],
    )


def test_implementation_hardware(exponential_filter_algo):
    num_adders, num_multipliers, memory_size = implementation_hardware(
        exponential_filter_algo
    )
    assert num_adders == 1
    assert num_multipliers == 2
    assert memory_size == 1

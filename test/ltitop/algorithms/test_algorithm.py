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
from ltitop.algorithms.statements import Assignment


@pytest.fixture
def cma_algo():
    x, n, cma = sympy.symbols("x n cma")

    return Algorithm(
        inputs=x,
        states=(n, cma),
        procedure=[
            Assignment(lhs=cma, rhs=(x + n * cma) / (n + 1)),
            Assignment(lhs=n, rhs=n + 1),
        ],
    )


def test_algorithm_execution(cma_algo):
    (n, cma), _ = cma_algo.collect(
        cma_algo.run(cma_algo.define(inputs=10, states=(1, 0)))
    )
    assert n == 2
    assert cma == 5

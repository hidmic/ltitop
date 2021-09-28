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

import mpmath
import pytest

from ltitop.arithmetic.rounding import ceil, floor, nearest_integer, truncate


@pytest.fixture(params=[float, mpmath.mpf])
def scalar(request):
    return request.param


def test_nearest_integer(scalar):
    assert scalar(-2) == nearest_integer.apply(scalar(-1.6))
    assert scalar(-1) == nearest_integer.apply(scalar(-1.4))
    assert scalar(0) == nearest_integer.apply(scalar(0))
    assert scalar(1) == nearest_integer.apply(scalar(1))
    assert scalar(1) == nearest_integer.apply(scalar(1.4))
    assert scalar(2) == nearest_integer.apply(scalar(1.6))


def test_floor(scalar):
    assert scalar(-2) == floor.apply(scalar(-1.6))
    assert scalar(-2) == floor.apply(scalar(-1.4))
    assert scalar(0) == floor.apply(scalar(0))
    assert scalar(1) == floor.apply(scalar(1))
    assert scalar(1) == floor.apply(scalar(1.4))
    assert scalar(1) == floor.apply(scalar(1.6))


def test_ceil(scalar):
    assert scalar(-1) == ceil.apply(scalar(-1.6))
    assert scalar(-1) == ceil.apply(scalar(-1.4))
    assert scalar(0) == ceil.apply(scalar(0))
    assert scalar(1) == ceil.apply(scalar(1))
    assert scalar(2) == ceil.apply(scalar(1.4))
    assert scalar(2) == ceil.apply(scalar(1.6))


def test_truncate(scalar):
    assert scalar(-1) == truncate.apply(scalar(-1.6))
    assert scalar(-1) == truncate.apply(scalar(-1.4))
    assert scalar(0) == truncate.apply(scalar(0))
    assert scalar(1) == truncate.apply(scalar(1))
    assert scalar(1) == truncate.apply(scalar(1.4))
    assert scalar(1) == truncate.apply(scalar(1.6))

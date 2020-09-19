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

from ltitop.arithmetic.rounding import ceil
from ltitop.arithmetic.rounding import floor
from ltitop.arithmetic.rounding import nearest_integer
from ltitop.arithmetic.rounding import truncate

import mpmath
import pytest

@pytest.fixture(params=[float, mpmath.mpf])
def scalar(request):
    return request.param

def test_nearest_integer(scalar):
    assert scalar(-2) == nearest_integer(scalar(-1.6))
    assert scalar(-1) == nearest_integer(scalar(-1.4))
    assert scalar(0) == nearest_integer(scalar(0))
    assert scalar(1) == nearest_integer(scalar(1))
    assert scalar(1) == nearest_integer(scalar(1.4))
    assert scalar(2) == nearest_integer(scalar(1.6))

def test_floor(scalar):
    assert scalar(-2) == floor(scalar(-1.6))
    assert scalar(-2) == floor(scalar(-1.4))
    assert scalar(0) == floor(scalar(0))
    assert scalar(1) == floor(scalar(1))
    assert scalar(1) == floor(scalar(1.4))
    assert scalar(1) == floor(scalar(1.6))

def test_ceil(scalar):
    assert scalar(-1) == ceil(scalar(-1.6))
    assert scalar(-1) == ceil(scalar(-1.4))
    assert scalar(0) == ceil(scalar(0))
    assert scalar(1) == ceil(scalar(1))
    assert scalar(2) == ceil(scalar(1.4))
    assert scalar(2) == ceil(scalar(1.6))

def test_truncate(scalar):
    assert scalar(-1) == truncate(scalar(-1.6))
    assert scalar(-1) == truncate(scalar(-1.4))
    assert scalar(0) == truncate(scalar(0))
    assert scalar(1) == truncate(scalar(1))
    assert scalar(1) == truncate(scalar(1.4))
    assert scalar(1) == truncate(scalar(1.6))

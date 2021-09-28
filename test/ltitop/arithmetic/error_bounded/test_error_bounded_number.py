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

import math

from ltitop.arithmetic.error_bounded import error_bounded
from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.rounding import nearest_integer


def test_error_bounded_addition():
    a = error_bounded(1.0, interval(-0.1, 0.3))
    b = error_bounded(1.0, interval(-0.3, 0.2))

    c = a + b
    assert c.number == 2.0
    assert math.isclose(c.error_bounds.lower_bound, -0.4)
    assert math.isclose(c.error_bounds.upper_bound, 0.5)

    d = a + 2
    assert d.number == 3.0
    assert d.error_bounds == a.error_bounds

    e = 1 + b
    assert e.number == 2.0
    assert e.error_bounds == b.error_bounds

    f = error_bounded(interval(-1, 1), interval(-0.1, 0.2))

    g = a + f
    assert g.number == interval(0, 2)
    assert g.error_bounds == interval(-0.2, 0.5)

    with FixedFormatArithmeticLogicUnit(format_=Q(7), rounding_method=nearest_integer):
        h = (a + (-0.5)) + fixed(0.2)
        assert h.number == fixed(0.7)
        assert a.error_bounds in h.error_bounds


def test_error_bounded_substraction():
    a = error_bounded(1.0, interval(-0.1, 0.3))
    b = error_bounded(1.0, interval(-0.3, 0.2))
    c = a - b
    assert c.number == 0.0
    assert math.isclose(c.error_bounds.lower_bound, -0.4)
    assert math.isclose(c.error_bounds.upper_bound, 0.5)

    d = a - 2
    assert d.number == -1.0
    assert d.error_bounds == a.error_bounds

    e = 1 - b
    assert e.number == 0.0
    assert e.error_bounds == b.error_bounds


def test_error_bounded_multiplication():
    a = error_bounded(0.5, interval(-0.1, 0.2))
    b = error_bounded(-4.0, interval(-0.2, 0.3))
    c = a * b
    assert c.number == -2.0
    assert math.isclose(c.error_bounds.lower_bound, -0.94)
    assert math.isclose(c.error_bounds.upper_bound, 0.52)

    d = a * 2
    assert d.number == 1.0
    assert math.isclose(d.error_bounds.lower_bound, -0.2)
    assert math.isclose(d.error_bounds.upper_bound, 0.4)

    e = 0.25 * b
    assert e.number == -1.0
    assert math.isclose(e.error_bounds.lower_bound, -0.05)
    assert math.isclose(e.error_bounds.upper_bound, 0.075)

    f = error_bounded(interval(-1, 1), interval(-0.1, 0.2))

    g = a * f
    assert g.number == interval(-0.5, 0.5)
    assert math.isclose(g.error_bounds.lower_bound, -0.27)
    assert math.isclose(g.error_bounds.upper_bound, 0.34)

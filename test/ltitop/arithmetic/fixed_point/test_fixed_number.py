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

from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.rounding import nearest_integer


def assert_close(actual, desired, atol):
    assert abs(mpfloat(actual) - mpfloat(desired)) <= atol


def assert_interval_close(actual, desired, atol):
    assert abs(mpfloat(actual).difference(mpfloat(desired))) <= atol


def test_fixed_numbers():
    with FixedFormatArithmeticLogicUnit(
        format_=Q(7), rounding_method=nearest_integer
    ) as alu:
        assert_interval_close(
            fixed(interval(-0.2, 0.1)), interval(-0.2, 0.1), atol=2 ** alu.format_.lsb
        )


def test_fixed_arithmetic():
    with FixedFormatArithmeticLogicUnit(
        format_=Q(7), rounding_method=nearest_integer
    ) as alu:
        assert_close(fixed(0.3) + fixed(0.2), 0.5, atol=2 ** alu.format_.lsb)
        assert_close(fixed(0.3) - fixed(0.2), 0.1, atol=2 ** alu.format_.lsb)
        assert_close(-fixed(0.3) + fixed(0.2), -0.1, atol=2 ** alu.format_.lsb)
        assert_close(fixed(0.3) * fixed(0.2), 0.06, atol=2 ** alu.format_.lsb)

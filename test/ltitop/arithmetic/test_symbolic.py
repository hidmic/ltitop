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

import sympy

from ltitop.arithmetic.error_bounded import error_bounded
from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q


def test_mixed_symbols():
    with FixedFormatArithmeticLogicUnit(format_=Q(7), allows_overflow=False):
        a = sympy.sympify(fixed(0.5))
        b = sympy.sympify(error_bounded(0.25))
        assert (a * b).number == fixed(0.125)

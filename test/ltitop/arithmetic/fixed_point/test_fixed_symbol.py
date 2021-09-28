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

import sympy

from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q
from ltitop.arithmetic.fixed_point.symbol import Fixed
from ltitop.arithmetic.rounding import nearest_integer


def test_literal_conversion():
    with FixedFormatArithmeticLogicUnit(format_=Q(7), rounding_method=nearest_integer):
        assert Fixed(0.25) == sympy.sympify(fixed(0.25))


def test_symbolic_expression():
    with FixedFormatArithmeticLogicUnit(
        format_=Q(7), allows_overflow=False, rounding_method=nearest_integer
    ) as alu:
        x, y, z, w = sympy.symbols("x y z w")
        expr = ((x + 0.5 * y) * z) - w
        subs = {x: fixed(0.25), y: fixed(0.1), z: 0.25, w: 0.5}
        result = expr.subs(subs)
        assert math.isclose(result, -0.425, abs_tol=2 ** alu.format_.lsb)

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

from ltitop.algorithms.expressions.arithmetic import associative
from ltitop.algorithms.expressions.arithmetic import nonassociative

from ltitop.arithmetic.rounding import nearest_integer
from ltitop.arithmetic.fixed_point.multi_format_arithmetic_logic_unit \
    import MultiFormatArithmeticLogicUnit
from ltitop.arithmetic.fixed_point.formats import uQ, Q
from ltitop.arithmetic.fixed_point import fixed


def test_nonassociative_expression():
    with MultiFormatArithmeticLogicUnit(
        wordlength=8,
        allows_overflow=False,
        allows_underflow=True,
        rounding_method=nearest_integer,
    ) as alu:
        x, y, z, w = sympy.symbols('x y z w')
        expr = x + y + z + w
        subs = {
            x: fixed(2**-5, format_=Q(2, 5)),
            y: fixed(2**-3, format_=Q(3, 4)),
            z: fixed(2**-1, format_=Q(4, 3)),
            w: fixed(2, format_=Q(5, 2))
        }
        result0 = nonassociative(variant=0)(expr).subs(subs)
        result1 = nonassociative(variant=1)(expr).subs(subs)
        assert math.isclose(result0, result1, abs_tol=2**result0.format_.lsb)
        assert result0 != result1

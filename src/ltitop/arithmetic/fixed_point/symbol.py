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

from sympy.core.numbers import Integer
from sympy.core.numbers import Float
from sympy.core.numbers import Rational
from ltitop.arithmetic.symbolic import BaseNumber
from ltitop.arithmetic.fixed_point.number import Number as FixedPointNumber


class Fixed(BaseNumber):
    __slots__ = ()

    is_real = True
    is_integer = False
    is_rational = True
    is_irrational = False

    # Ensure fixed point operators take precedence
    _op_priority = BaseNumber._op_priority * 20

    def __new__(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Integer):
            value = value.p
        elif isinstance(value, (Float, Rational)):
            value = mpmath.make_mpf(
                value._as_mpf_val(mpmath.mp.prec)
            )
        if not isinstance(value, FixedPointNumber):
            value = FixedPointNumber.from_value(value)
        return cls._new(value)

    def _eval_is_finite(self):
        return True

    def _eval_is_infinite(self):
        return False

    @property
    def p(self):
        return float(self._args[0])

    @property
    def q(self):
        return 1

from sympy.core.sympify import converter
converter[FixedPointNumber] = Fixed

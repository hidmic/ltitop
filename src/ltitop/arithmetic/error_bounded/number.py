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

from typing import Any

from ltitop.arithmetic.interval import Interval
from ltitop.arithmetic.fixed_point.number import Number as FixedPointNumber
from ltitop.arithmetic.fixed_point.symbol import Fixed as FixedPointSymbol
from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.rounding import nearest_integer, ceil, floor, truncate
from ltitop.common.dataclasses import immutable_dataclass


@immutable_dataclass
class Number:
    number: Any
    error_bounds: Interval = Interval(mpfloat(0))

    def __add__(self, other):
        if not isinstance(other, Number):
            other = Number(other)
        result = self.number + other.number
        result_error_bounds = self.error_bounds + other.error_bounds
        if isinstance(result, (FixedPointNumber, FixedPointSymbol)):
            exact = self.number == 0 or other.number == 0
            if not exact:
                rounding = ProcessingUnit.active().rounding_method
                if isinstance(self.number, (FixedPointNumber, FixedPointSymbol)):
                    if result.format_.lsb > self.number.format_.lsb:
                        result_error_bounds += rounding.error_bounds(
                            result.format_.lsb, self.number.format_.lsb
                        )
                elif self.number != 0:
                    result_error_bounds += rounding.error_bounds(result.format_.lsb)
                if isinstance(other.number, (FixedPointNumber, FixedPointSymbol)):
                    if result.format_.lsb > other.number.format_.lsb:
                        result_error_bounds += rounding.error_bounds(
                            result.format_.lsb, other.number.format_.lsb
                        )
                elif other.number != 0:
                    result_error_bounds += rounding.error_bounds(result.format_.lsb)
        return Number(result, result_error_bounds)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Number):
            other = Number(other)
        result = self.number - other.number
        result_error_bounds = self.error_bounds + other.error_bounds
        if isinstance(result, (FixedPointNumber, FixedPointSymbol)):
            exact = self.number == 0 or other.number == 0
            if not exact:
                rounding = ProcessingUnit.active().rounding_method
                if isinstance(self.number, (FixedPointNumber, FixedPointSymbol)):
                    if result.format_.lsb > self.number.format_.lsb:
                        result_error_bounds += rounding.error_bounds(
                            result.format_.lsb, self.number.format_.lsb
                        )
                elif self.number != 0:
                    result_error_bounds += rounding.error_bounds(result.format_.lsb)
                if isinstance(other.number, (FixedPointNumber, FixedPointSymbol)):
                    if result.format_.lsb > other.number.format_.lsb:
                        result_error_bounds += rounding.error_bounds(
                            result.format_.lsb, other.number.format_.lsb
                        )
                elif other.number != 0:
                    result_error_bounds += rounding.error_bounds(result.format_.lsb)
        return Number(result, result_error_bounds)

    def __rsub__(self, other):
        return Number(other) - self

    def __mul__(self, other):
        if not isinstance(other, Number):
            other = Number(other)
        result = self.number * other.number
        snumber = mpfloat(self.number)
        onumber = mpfloat(other.number)
        result_error_bounds = (
            (snumber + self.error_bounds) *
            (onumber + other.error_bounds)
        ).difference(snumber * onumber)
        if isinstance(result, (FixedPointNumber, FixedPointSymbol)):
            exact = (self.number == -1 or self.number == 1 or self.number == 0 or
                     other.number == -1 or other.number == 1 or other.number == 0)
            if not exact:
                rounding = ProcessingUnit.active().rounding_method
                if isinstance(self.number, (FixedPointNumber, FixedPointSymbol)) and \
                   isinstance(other.number, (FixedPointNumber, FixedPointSymbol)):
                    result_error_bounds += rounding.error_bounds(
                        result.format_.lsb,
                        self.number.format_.lsb + other.number.format_.lsb
                    )
                else:
                    result_error_bounds += rounding.error_bounds(result.format_.lsb)
        return Number(result, result_error_bounds)

    __rmul__ = __mul__

    def __div__(self, other):
        # TODO(hidmic): implement
        return NotImplemented

    __truediv__ = __div__

    def __rdiv__(self, other):
        return Number(other) / self

    __rtruediv__ = __rdiv__

    def __mod__(self, other):
        # TODO(hidmic): implement
        return NotImplemented

    def __rmod__(self, other):
        return Number(other) % self

    def __apply_rounding_method__(self, method):
        number = method(self.number)
        error_bounds = self.error_bounds
        lsb = None
        if isinstance(self.number, (FixedPointNumber, FixedPointSymbol)):
            lsb = self.number.format_.lsb
        error_bounds += method.error_bounds(0, lsb)
        return Number(number, error_bounds)

    def __trunc__(self):
        return self.__apply_rounding_method__(truncate)

    def __ceil__(self):
        return self.__apply_rounding_method__(ceil)

    def __floor__(self):
        return self.__apply_rounding_method__(floor)

    def __round__(self):
        return self.__apply_rounding_method__(nearest_integer)

    def __neg__(self):
        return Number(-self.number, -self.error_bounds)

    def __float__(self):
        return float(self.number)

    def __mpfloat__(self):
        return mpfloat(self.number)

    def __int__(self):
        return int(self.number)

    __long__ = __int__

    def __nonzero__(self):
        return bool(self.number)

    __bool__ = __nonzero__

    _iterable = False  # tell sympy to not iterate this

    def __getitem__(self, key):
        try:
            error_bounds = self.error_bounds[key]
        except TypeError:
            error_bounds = self.error_bounds
        return Number(self.number[key], error_bounds)

    def __eq__(self, other):
        if not isinstance(other, Number):
            other = Number(other)
        return self.number == other.number and \
            self.error_bounds == other.error_bounds == 0

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if not isinstance(other, Number):
            other = Number(other)
        return mpfloat(self.number) + self.error_bounds < \
            mpfloat(other.number) + other.error_bounds

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def __lshift__(self, n):
        return Number(self.number << n, self.error_bounds * 2**n)

    def __rshift__(self, n):
        # TODO(hidmic): implement
        return NotImplemented

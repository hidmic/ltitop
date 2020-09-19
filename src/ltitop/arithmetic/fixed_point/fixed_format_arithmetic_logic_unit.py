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

import functools
import numpy as np

from ltitop.arithmetic.errors import OverflowError
from ltitop.arithmetic.errors import UnderflowError
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.fixed_point.formats import Format
from ltitop.arithmetic.fixed_point.representation import Representation
from ltitop.arithmetic.rounding import nearest_integer

from ltitop.arithmetic.fixed_point.arithmetic_logic_unit import ArithmeticLogicUnit
from ltitop.arithmetic.floating_point import mpfloat

from ltitop.common.memoization import memoize


class FixedFormatArithmeticLogicUnit(ArithmeticLogicUnit):

    class internals:
        @staticmethod
        def operation_method(method):
            @functools.wraps(method)
            def __wrapper(self, head, *tail):
                if __debug__:
                    if head.format_ != self.format_:
                        raise ValueError(f'{self} cannot handle {head}')
                    for op in tail:
                        if op.format_ != self.format_:
                            raise ValueError(f'{self} cannot handle {op}')
                return method(self, head, *tail)
            return __wrapper

    def __init__(self, *, format_, **kwargs):
        super().__init__(wordlength=format_.wordlength, **kwargs)
        self.__format = format_

    @property
    def format_(self):
        return self.__format

    @functools.lru_cache(maxsize=128)
    def represent(self, value, rtype=Representation):
        if isinstance(value, rtype) and value.format_ == self.format_:
            return value
        mantissa, (underflow, overflow) = self.format_.represent(
            value, rounding_method=nearest_integer
        )
        if underflow and not self.represent.allows_underflow:
            raise UnderflowError(
                f'{value} underflows in {self.format_}',
                value, self.format_.value_epsilon
            )
        if overflow:
            if not self.represent.allows_overflow:
                raise OverflowError(
                    f'{value} overflows in {self.format_}',
                    value, self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return rtype(mantissa, self.format_)

    @memoize
    def rinfo(self):
        return FixedFormatArithmeticLogicUnit.Info(
            eps=self.format_.value_epsilon,
            min=self.format_.value_interval.lower_bound,
            max=self.format_.value_interval.upper_bound
        )

    @internals.operation_method
    def add(self, x, y):
        mantissa = x.mantissa + y.mantissa
        if self.format_.overflows_with(mantissa):
            if not self.add.allows_overflow:
                raise OverflowError(
                    f'{x} + {y} overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, self.format_)

    @internals.operation_method
    def substract(self, x, y):
        mantissa = x.mantissa - y.mantissa
        if self.format_.overflows_with(mantissa):
            if not self.substract.allows_overflow:
                raise OverflowError(
                    f'{x} - {y} overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, self.format_)

    @internals.operation_method
    def multiply(self, x, y):
        # Use 2 * wordlength long multipliers
        z = Representation(
            mantissa=x.mantissa * y.mantissa,
            format_=Format(
                msb=self.format_.msb * 2 + 1,
                lsb=self.format_.lsb * 2,
                signed=self.format_.signed
            )
        )
        mantissa, (underflow, overflow) = self.format_.represent(
            z, rounding_method=self.rounding_method
        )
        if underflow and not self.multiply.allows_underflow:
            raise UnderflowError(
                f'{x} * {y} underflows in {self.format_}',
                mpfloat(z), self.format_.value_epsilon
            )
        if overflow:
            if not self.multiply.allows_overflow:
                raise OverflowError(
                    f'{x} * {y} overflows in {self.format_}',
                    mpfloat(z), self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, self.format_)

    @internals.operation_method
    def truncate(self, x):
        if x.is_integer:
            return x
        value = truncate(mpfloat(x))
        mantissa, (_, overflow) = self.format_.represent(value)
        if overflow:
            if not self.truncate.allows_overflow:
                raise OverflowError(
                    f'trunc({x}) overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, format_)

    @internals.operation_method
    def floor(self, x):
        if x.is_integer:
            return x
        value = floor(mpfloat(x))
        mantissa, (_, overflow) = self.format_.represent(value)
        if overflow:
            if not self.floor.allows_overflow:
                raise OverflowError(
                    f'floor({x}) overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, format_)

    @internals.operation_method
    def ceil(self, x):
        if x.is_integer:
            return x
        value = ceil(mpfloat(x))
        mantissa, (_, overflow) = self.format_.represent(value)
        if overflow:
            if not self.ceil.allows_overflow:
                raise OverflowError(
                    f'ceil({x}) overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, format_)

    @internals.operation_method
    def nearest(self, x):
        if x.is_integer:
            return x
        value = nearest_integer(mpfloat(x))
        mantissa, (_, overflow) = self.format_.represent(value)
        if overflow:
            if not self.nearest.allows_overflow:
                raise OverflowError(
                    f'round({x}) overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, format_)

    @internals.operation_method
    def negate(self, x):
        if not self.format_.signed:
            raise RuntimeError('Cannot negate unsigned representations')
        if x.mantissa == 0:
            return x
        mantissa = -x.mantissa
        if self.format_.overflows_with(mantissa):
            if not self.negate.allows_overflow:
                raise OverflowError(
                    f'{x} overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, self.format_)

    @internals.operation_method
    def compare(self, x, y):
        return x.mantissa - y.mantissa

    def lshift(self, x, n):
        if x.format_ != self.format_:
            raise ValueError(f'{self} cannot handle {x}')
        if n < 0:
            raise ValueError(f'negative shift count {n}')
        n = int(n)
        mantissa = x.mantissa << n
        if self.format_.overflows_with(mantissa):
            if not self.lshift.allows_overflow:
                raise OverflowError(
                    f'{x} << {n} overflows in {self.format_}',
                    mantissa * self.format_.value_epsilon,
                    self.format_.value_interval
                )
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=self.format_.mantissa_interval
            )
        return type(x)(mantissa, self.format_)

    def rshift(self, x, n):
        if x.format_ != self.format_:
            raise ValueError(f'{self} cannot handle {x}')
        if n < 0:
            raise ValueError(f'negative shift count {n}')
        n = int(n)
        return type(x)(x.mantissa >> n, self.format_)

    def __str__(self):
        return f'{self.format_.to_qnotation()} ALU'

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
import mpmath
import numpy as np

from ltitop.arithmetic.errors import UnderflowError
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.fixed_point.formats import Format
from ltitop.arithmetic.fixed_point.representation import Representation

from ltitop.arithmetic.fixed_point.arithmetic_logic_unit import ArithmeticLogicUnit
from ltitop.arithmetic.floating_point import mpmsb

from ltitop.common.memoization import memoize


class MultiFormatArithmeticLogicUnit(ArithmeticLogicUnit):

    class internals:
        @staticmethod
        def operation_method(method):
            @functools.wraps(method)
            def __wrapper(self, head, *tail):
                if __debug__:
                    if head.format_.wordlength > self.wordlength:
                        raise ValueError(f'{self} cannot handle {head}')
                    signed = head.format_.signed
                    for op in tail:
                        if op.format_.wordlength > self.wordlength:
                            raise ValueError(f'{self} cannot handle {op}')
                        if signed != op.format_.signed:
                            raise ValueError(f'{self} cannot handle mixed signs')
                return method(self, head, *tail)
            return __wrapper

    @functools.lru_cache(maxsize=128)
    def represent(self, value, rtype=Representation, format_=None):
        if format_ is not None:
            if format_.wordlength > self.wordlength:
                raise ValueError(f'{format_} wordlength is too large')
            mantissa, (underflow, overflow) = format_.represent(
                value, rounding_method=self.rounding_method
            )
            if underflow and not self.represent.allows_underflow:
                raise UnderflowError(f'{value} in {format_} underflows')
            if overflow:
                if not self.represent.allows_overflow:
                    raise OverflowError(f'{value} in {format_} overflows')
                mantissa, _ = self.overflow_behavior(
                    mantissa, range_=format_.mantissa_interval
                )
        else:
            if isinstance(value, rtype):
                if value.format_.wordlength <= self.wordlength:
                    return value
            mantissa, format_ = Format.best(value, wordlength=self.wordlength)
        return rtype(mantissa, format_)

    @memoize
    def rinfo(self, *, signed=True):
        if signed:
            epsilon = mpmath.ldexp(1, -self.wordlength + 1)
            limit = mpmath.ldexp(1, self.wordlength - 1)
            return MultiFormatArithmeticLogicUnit.Info(
                eps=epsilon, min=-limit, max=limit - 1
            )
        epsilon = mpmath.ldexp(1, -self.wordlength)
        limit = mpmath.ldexp(1, self.wordlength)
        return MultiFormatArithmeticLogicUnit.Info(
            eps=epsilon, min=mpmath.mp.zero, max=limit
        )

    def _find_common_format(self, format_x, format_y):
        if format_x == format_y:
            return format_x
        assert format_x.signed == format_y.signed
        signed = format_x.signed
        msb = max(format_x.msb, format_y.msb)
        lsb = min(format_x.lsb, format_y.lsb)
        wordlength = msb - lsb + int(signed)
        if wordlength > self.wordlength:
            lsb = msb - self.wordlength + int(signed)
        return Format(msb, lsb, signed)

    def _align_operand_mantissa(self, value, format_, allows_underflow):
        if value.format_ == format_:
            return value.mantissa
        mantissa, (underflow, overflow) = format_.represent(
            value, rounding_method=self.rounding_method
        )
        if underflow and not allows_underflow:
            raise UnderflowError(f'{value} underflows in {format_}')
        assert not overflow
        return mantissa

    def _handle_overflow(self, operation, mantissa, format_, off_by):
        if not operation.allows_overflow:
            signed = format_.signed
            msb = format_.msb + off_by
            extended_format = Format(msb, format_.lsb, signed)
            lsb = msb - self.wordlength + int(signed)
            format_ = Format(msb, lsb, signed)
            mantissa, (underflow, overflow) = format_.represent(
                Representation(mantissa, extended_format),
                rounding_method=self.rounding_method
            )
            assert not underflow
            assert not overflow
        else:
            mantissa, _ = self.overflow_behavior(
                mantissa, range_=format_.mantissa_interval
            )
        return mantissa, format_

    @internals.operation_method
    def add(self, x, y):
        format_z = self._find_common_format(x.format_, y.format_)
        mantissa_x = self._align_operand_mantissa(
            x, format_z, allows_underflow=self.add.allows_underflow)
        mantissa_y = self._align_operand_mantissa(
            y, format_z, allows_underflow=self.add.allows_underflow)
        mantissa_z = mantissa_x + mantissa_y
        if format_z.overflows_with(mantissa_z):
            mantissa_z, format_z = self._handle_overflow(
                self.add, mantissa_z, format_z, off_by=1)
        return type(x)(mantissa_z, format_z)

    @internals.operation_method
    def substract(self, x, y):
        # Use 1x wordlength adders with carry
        format_z = self._find_common_format(x.format_, y.format_)
        mantissa_x, underflow = self._align_operand_mantissa(
            x, format_z, allows_underflow=self.substract.allows_underflow)
        mantissa_y, underflow = self._align_operand_mantissa(
            y, format_z, allows_underflow=self.substract.allows_underflow)
        mantissa_z = mantissa_x - mantissa_y
        if format_z.overflows_with(mantissa_z):
            if not self.substract.allows_overflow and not format_z.signed:
                raise OverflowError(f'Cannot prevent unsigned overflow')
            mantissa_z, format_z = self._handle_overflow(
                self.substract, mantissa_z, format_z, off_by=1)
        return type(x)(mantissa_z, format_z)

    @internals.operation_method
    def multiply(self, x, y):
        # Use 2x wordlength multipliers, adjust format
        signed = x.format_.signed
        lsb = x.format_.lsb + y.format_.lsb
        msb = x.format_.msb + y.format_.msb
        format_z = Format(msb, lsb, signed)
        mantissa_z = x.mantissa * y.mantissa
        if format_z.wordlength > self.wordlength:
            mantissa_z, format_z = Format.best(
                Representation(mantissa_z, format_z),
                wordlength=self.wordlength,
                rounding_method=self.rounding_method,
                signed=signed
            )
        return type(x)(mantissa_z, format_z)

    @internals.operation_method
    def truncate(self, x):
        if x.is_integer:
            return x
        format_ = x.format_
        value = truncate(mpfloat(x))
        mantissa, (_, overflow) = format_.represent(value)
        assert not overflow
        return type(x)(mantissa, x.format_)

    @internals.operation_method
    def floor(self, x):
        if x.is_integer:
            return x
        format_ = x.format_
        mantissa, (_, overflow) = format_.represent(floor(mpfloat(x)))
        if overflow:
            mantissa, format_ = self._handle_overflow(
                self.floor, mantissa, format_, off_by=1)
        return type(x)(mantissa, format_)

    @internals.operation_method
    def ceil(self, x):
        if x.is_integer:
            return x
        format_ = x.format_
        mantissa, (_, overflow) = format_.represent(ceil(mpfloat(x)))
        if overflow:
            mantissa, format_ = self._handle_overflow(
                self.ceil, mantissa, format_, off_by=1)
        return type(x)(mantissa, format_)

    @internals.operation_method
    def nearest(self, x):
        if x.is_integer:
            return x
        format_ = x.format_
        mantissa, (_, overflow) = format_.represent(nearest_integer(mpfloat(x)))
        if overflow:
            mantissa, format_ = self._handle_overflow(
                self.nearest, mantissa, format_, off_by=1)
        return type(x)(mantissa, format_)

    @internals.operation_method
    def negate(self, x):
        if not x.format_.signed:
            raise ValueError(f'Cannot negate unsigned representation {x}')
        if x.mantissa == 0:
            return x
        format_ = x.format_
        mantissa = -x.mantissa
        if format_.overflows_with(mantissa):
            mantissa, format_ = self._handle_overflow(
                self.negate, mantissa, format_, off_by=1)
        return type(x)(mantissa, format_)

    @internals.operation_method
    def compare(self, x, y):
        format_ = self._find_common_format(x.format_, y.format_)
        mantissa_x = self._align_operand_mantissa(
            x, format_, allows_underflow=self.compare.allows_underflow)
        mantissa_y = self._align_operand_mantissa(
            y, format_, allows_underflow=self.compare.allows_underflow)
        return mantissa_x - mantissa_y

    def lshift(self, x, n):
        if x.format_.wordlength > self.wordlength:
            raise ValueError(f'{self} cannot handle {x}')
        if n < 0:
            raise ValueError(f'negative shift count {n}')
        n = int(n)
        return type(x)(x.mantissa, Format(
            msb=x.format_.msb + n,
            lsb=x.format_.lsb + n,
            signed=x.format_.signed
        ))

    def rshift(self, x, n):
        if x.format_.wordlength > self.wordlength:
            raise ValueError(f'{self} cannot handle {x}')
        if n < 0:
            raise ValueError(f'negative shift count {n}')
        n = int(n)
        return type(x)(x.mantissa, Format(
            msb=x.format_.msb - n,
            lsb=x.format_.lsb - n,
            signed=x.format_.signed
        ))

    def __str__(self):
        return f'{self.wordlength} bits multi-format ALU'

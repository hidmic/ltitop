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

import re
import mpmath
import functools
import numpy as np

from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.floating_point import mpmsb
from ltitop.arithmetic.floating_point import mpquantize
from ltitop.arithmetic.rounding import nearest_integer
from ltitop.arithmetic.rounding import floor
from ltitop.arithmetic.rounding import ceil
from ltitop.arithmetic.rounding import truncate
from ltitop.arithmetic.modular import wraparound
from ltitop.arithmetic.interval import interval
from ltitop.common.memoization import memoize
from ltitop.common.dataclasses import immutable_dataclass


@immutable_dataclass
class Format:
    msb: int
    lsb: int
    signed: bool = True

    _qnotation_pattern = \
        re.compile(r"^([su]?)Q([+-]?[0-9]+)\.([+-]?[0-9]+)$")

    @classmethod
    def from_qnotation(cls, notation):
        match = cls._qnotation_pattern.match(notation)
        if not match:
            raise ValueError("'{}' is not in Q notation".format(notation))
        signed, msb, lsb = match.groups()
        signed = (signed != 'u')
        msb = int(msb) - (1 if signed else 0)
        lsb = -int(lsb)
        return cls(msb=msb, lsb=lsb, signed=signed)

    _pnotation_pattern = \
        re.compile(r"^([su]?)\(([+-]?[0-9]+),([+-]?[0-9]+)\)$")

    @classmethod
    def from_pnotation(cls, notation):
        match = cls._pnotation_pattern.match(notation)
        if not match:
            raise ValueError(
                "'{}' is not in parenthesis notation".format(notation)
            )
        signed, msb, lsb = match.groups()
        signed = (signed != 'u')
        msb = int(msb)
        lsb = int(lsb)
        return cls(msb=msb, lsb=lsb, signed=signed)

    @classmethod
    def from_notation(cls, notation):
        try:
            return cls.from_qnotation(notation)
        except ValueError:
            pass
        try:
            return cls.from_pnotation(notation)
        except ValueError:
            pass
        raise ValueError(f"'{notation}' is not in a known notation")

    @classmethod
    def best(cls, value, wordlength, rounding_method=nearest_integer, signed=True):
        """
        See "Reliable Implementation of Linear Filters with Fixed-Point Arithmetic",
        Hilaire and Lopez, 2013
        """
        # TODO(hidmic): optimize when value is a Representation
        with mpmath.workprec(20 * wordlength):  # enough bits
            mpvalue = mpfloat(value)
            if not np.all(mpvalue >= 0) and not signed:
                raise ValueError(f'Unsigned format cannot represent {value}')
            # estimate MSB
            msb = np.max(mpmsb(mpvalue, signed=signed))
            if np.isneginf(msb):
                msb = 0  # arbitrary
            msb = int(msb)
            # estimate LSB
            lsb = msb - wordlength + int(signed)
            # compute mantissa
            mantissa = mpquantize(
                mpvalue, nbits=-lsb,
                rounding_method=rounding_method
            )
            # adjust MSB if limits were overpassed
            adjusted_msb = msb  # no adjustment
            upper_limit = 2 ** (wordlength - int(signed))
            if not np.all(mantissa < upper_limit):
                adjusted_msb = msb + 1
            if np.all(mantissa < 0):
                lower_limit = -2 ** (wordlength - 2)
                if np.all(mantissa > lower_limit):
                    adjusted_msb = msb - 1
            if adjusted_msb != msb:
                # adjust LSB and mantissa
                msb = adjusted_msb
                lsb = msb - wordlength + int(signed)
                mantissa = mpquantize(
                    mpvalue, nbits=-lsb,
                    rounding_method=mpmath.nint
                )
            return mantissa, cls(msb=msb, lsb=lsb, signed=signed)

    @classmethod
    def Q(cls, a, b=None):
        if b is None:
            b = a
            a = 1
        return cls(msb=a-1, lsb=-b, signed=True)

    sQ = Q

    @classmethod
    def uQ(cls, a, b=None):
        if b is None:
            b = a
            a = 1
        return cls(msb=a, lsb=-b, signed=False)

    @classmethod
    def P(cls, a, b):
        return cls(msb=a, lsb=b, signed=True)

    sP = P

    @classmethod
    def uP(cls, a, b):
        return cls(msb=a, lsb=b, signed=False)

    def __post_init__(self):
        if self.lsb > self.msb:
            raise ValueError(
                'Least significant bit (LSB) cannot be larger than'
                f' most significant bit (MSB): {self.lsb} > {self.msb}'
            )

    @property
    @memoize
    def wordlength(self):
        return self.msb - self.lsb + int(self.signed)

    @property
    @memoize
    def mantissa_interval(self):
        if self.signed:
            return interval(
                lower_bound=-2**(self.wordlength - 1),
                upper_bound=2**(self.wordlength - 1) - 1
            )
        return interval(lower_bound=0, upper_bound=2**self.wordlength - 1)

    def overflows_with(self, mantissa):
        return bool(np.any(mantissa not in self.mantissa_interval))

    @property
    @memoize
    def value_interval(self):
        with mpmath.workprec(self.wordlength + 1):  # wl bits is enough
            if self.signed:
                return interval(
                    -mpmath.ldexp(1, self.msb),
                    mpmath.ldexp(1, self.msb) - mpmath.ldexp(1, self.lsb)
                )
            return interval(0, mpmath.ldexp(1, self.msb) - mpmath.ldexp(1, self.lsb))

    @property
    @memoize
    def value_epsilon(self):
        return mpmath.ldexp(1, self.lsb)

    def can_represent(self, value):
        return bool(np.all(value in self.value_interval))

    def __eq__(self, other):
        return self.msb == other.msb and self.lsb == other.lsb and self.signed == other.signed

    def represent(self, rvalue, rounding_method=round):
        from ltitop.arithmetic.fixed_point.representation import Representation
        if isinstance(rvalue, Representation) and hasattr(rounding_method, 'shift'):
            lvalue = rvalue.mantissa
            quantize = functools.partial(
                rounding_method.shift,
                n=rvalue.format_.lsb - self.lsb)
        else:
            lvalue = mpfloat(rvalue)
            quantize = functools.partial(
                mpquantize, nbits=-self.lsb,
                rounding_method=rounding_method)
        if not self.signed and not np.all(lvalue >= 0):
            raise ValueError(f'Unsigned format cannot represent {rvalue}')
        mantissa = quantize(lvalue)
        underflow = bool(np.any(np.logical_and(mantissa == 0, lvalue != 0)))
        overflow = bool(np.any(mantissa not in self.mantissa_interval))
        return mantissa, (underflow, overflow)

    def to_qnotation(self):
        notation = "Q{}.{}".format(
            self.msb + (1 if self.signed else 0), -self.lsb
        )
        return 'u' + notation if not self.signed else notation

    def to_pnotation(self):
        notation = "({},{})".format(self.msb, self.lsb)
        return 'u' + notation if not self.signed else notation

Q = Format.Q
sQ = Format.sQ
uQ = Format.uQ

P = Format.P
sP = Format.sP
uP = Format.uP

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
import numpy as np

from ltitop.arithmetic.interval import Interval
from ltitop.common.arrays import vectorize


@vectorize
def mpfloat(value):
    try:
        return value.astype(mpfloat)
    except (AttributeError, TypeError):
        return mpmath.mpmathify(value)


@vectorize(excluded=["signed"])
def mpmsb(value, signed):
    if isinstance(value, Interval):
        return np.max(
            [
                mpmsb(value.lower_bound, signed=signed),
                mpmsb(value.upper_bound, signed=signed),
            ]
        )
    if value > 0:
        return int(mpmath.floor(mpmath.log(value, 2))) + int(signed)
    if value < 0:
        return int(mpmath.ceil(mpmath.log(-value, 2)))
    return -np.inf


@vectorize(excluded=["nbits", "rounding_method"])
def mpquantize(value, nbits, rounding_method):
    if isinstance(value, Interval):
        return Interval(
            lower_bound=mpquantize(
                value.lower_bound, nbits=nbits, rounding_method=rounding_method
            ),
            upper_bound=mpquantize(
                value.upper_bound, nbits=nbits, rounding_method=rounding_method
            ),
        )
    if not mpmath.isfinite(value):
        raise ValueError(f"Cannot quantize non-finite value: {value}")
    return int(rounding_method.apply(mpmath.ldexp(value, int(nbits))))

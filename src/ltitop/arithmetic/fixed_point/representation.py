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

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ltitop.arithmetic.fixed_point.formats import Format
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.interval import Interval
from ltitop.common.dataclasses import immutable_dataclass


@immutable_dataclass
class Representation:
    mantissa: Union[int, Interval, ArrayLike]
    format_: Format

    def __post_init__(self):
        if __debug__:
            if np.any(self.mantissa not in self.format_.mantissa_interval):
                raise ValueError(
                    f"{self.astype(float)} cannot be " f"represented in {self.format_}"
                )

    @property
    def is_integer(self):
        return self.format_.lsb >= 0

    def astype(self, dtype):
        try:
            mantissa = self.mantissa.astype(dtype)
        except (AttributeError, TypeError):
            mantissa = dtype(self.mantissa)
        return mantissa * 2 ** self.format_.lsb

    def __float__(self):
        return self.astype(float)

    def __mpfloat__(self):
        return self.astype(mpfloat)

    def __int__(self):
        if self.format_.lsb < 0:
            return self.mantissa >> -self.format_.lsb
        return self.mantissa << self.format_.lsb

    __long__ = __int__

    def __nonzero__(self):
        if not isinstance(self.mantissa, np.ndarray):
            return self.mantissa != 0
        return bool(np.any(self.mantissa != 0))

    __bool__ = __nonzero__

    _iterable = False  # tell sympy to not iterate this

    def __getitem__(self, key):
        return type(self)(self.mantissa[key], self.format_)

    def __hash__(self):
        mantissa = self.mantissa
        if hasattr(mantissa, "tobytes"):
            mantissa = mantissa.tobytes()
        return hash((mantissa, self.format_))

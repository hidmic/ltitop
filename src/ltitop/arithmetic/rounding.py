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

import numpy as np

from ltitop.arithmetic.interval import Interval, interval


class nearest_integer:
    @staticmethod
    def apply(x):
        return round(x)

    @staticmethod
    def shift(value, n):
        if isinstance(value, Interval):
            return Interval(
                lower_bound=nearest_integer.shift(value.lower_bound, n),
                upper_bound=nearest_integer.shift(value.upper_bound, n),
            )
        result = floor.shift(value, n)
        if n < 0:
            result += (value >> (-n - 1)) % 2
        return result

    @staticmethod
    def error_bounds(output_lsb, input_lsb=None):
        if input_lsb is None:
            return interval(-(2 ** (output_lsb - 1)), 2 ** (output_lsb - 1))
        return interval(
            -(2 ** (output_lsb - 1)) + 2 ** input_lsb, 2 ** (output_lsb - 1)
        )


class floor:
    @staticmethod
    def apply(x):
        return math.floor(x)

    @staticmethod
    def error_bounds(output_lsb, input_lsb=None):
        if input_lsb is None:
            return interval(-(2 ** output_lsb), 0)
        return interval(-(2 ** output_lsb) + 2 ** input_lsb, 0)

    @staticmethod
    def shift(value, n):
        if n > 0:
            return value << n
        if n < 0:
            return value >> -n
        return value


class ceil:
    @staticmethod
    def apply(x):
        return math.ceil(x)

    @staticmethod
    def shift(value, n):
        return -floor.shift(-value, n)

    @staticmethod
    def error_bounds(output_lsb, input_lsb=None):
        if input_lsb is None:
            return interval(0, 2 ** output_lsb)
        return interval(0, 2 ** output_lsb - 2 ** input_lsb)


class truncate:
    @staticmethod
    def apply(x):
        try:
            return math.trunc(x)
        except TypeError:
            return math.floor(x) if x > 0 else math.ceil(x)

    @staticmethod
    def shift(value, n):
        if isinstance(value, Interval):
            return Interval(
                lower_bound=truncate.shift(value.lower_bound, n),
                upper_bound=truncate.shift(value.upper_bound, n),
            )
        results = np.array([floor.shift(value, n), ceil.shift(value, n)])
        return results[np.argmin(np.abs(results), axis=0), np.arange(results.shape[1])]

    @staticmethod
    def error_bounds(output_lsb, input_lsb=None):
        bounds = interval(-(2 ** output_lsb), 2 ** output_lsb)
        if input_lsb is not None:
            bounds += interval(2 ** input_lsb, -(2 ** input_lsb))
        return bounds

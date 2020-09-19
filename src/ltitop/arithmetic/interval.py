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

import collections
import typing
import mpmath
import numpy as np

from ltitop.common.arrays import asscalar_if_possible
from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.helpers import astuple


@immutable_dataclass
class Interval:
    lower_bound: typing.Any
    upper_bound: typing.Any = None

    def __post_init__(self):
        if self.upper_bound is None:
            try:
                if len(self.lower_bound) != 2:
                    raise ValueError('Intervals can only be initialized from tuples')
                super().__setattr__('upper_bound', self.lower_bound[1])
                super().__setattr__('lower_bound', self.lower_bound[0])
            except TypeError:  # if len() fails
                super().__setattr__('upper_bound', self.lower_bound)
        if __debug__:
            if np.any(self.upper_bound < self.lower_bound):
                raise ValueError(
                    f'Interval upper bound {self.upper_bound} cannot'
                    f' be lower than lower bound {self.lower_bound}')

    def __abs__(self):
        lower_bound = np.max([
            self.lower_bound,
            np.zeros_like(self.lower_bound),
        ], axis=0)
        upper_bound = np.max([
            np.abs(self.lower_bound),
            np.abs(self.upper_bound)
        ], axis=0)
        return Interval(lower_bound, upper_bound)

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower_bound + other.lower_bound,
                            self.upper_bound + other.upper_bound)
        return Interval(self.lower_bound + other, self.upper_bound + other)

    def __radd__(self, other):
        return Interval(other + self.lower_bound, other + self.upper_bound)

    def difference(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower_bound - other.lower_bound,
                            self.upper_bound - other.upper_bound)
        return Interval(self.lower_bound - other, self.upper_bound - other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower_bound - other.upper_bound,
                            self.upper_bound - other.lower_bound)
        return Interval(self.lower_bound - other, self.upper_bound - other)

    def __rsub__(self, other):
        return Interval(other - self.lower_bound, other - self.upper_bound)

    def __mul__(self, other):
        if isinstance(other, Interval):
            a = self.lower_bound * other.lower_bound
            b = self.lower_bound * other.upper_bound
            c = self.upper_bound * other.lower_bound
            d = self.upper_bound * other.upper_bound
            return Interval(np.min([a, b, c, d], axis=0),
                            np.max([a, b, c, d], axis=0))
        a = self.lower_bound * other
        b = self.upper_bound * other
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __rmul__(self, other):
        a = other * self.lower_bound
        b = other * self.upper_bound
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __div__(self, other):
        if isinstance(other, Interval):
            a = self.lower_bound / other.lower_bound
            b = self.lower_bound / other.upper_bound
            c = self.upper_bound / other.lower_bound
            d = self.upper_bound / other.upper_bound
            return Interval(np.min([a, b, c, d], axis=0),
                            np.max([a, b, c, d], axis=0))
        a = self.lower_bound / other
        b = self.upper_bound / other
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))


    __truediv__ = __div__

    def __rdiv__(self, other):
        a = other / self.lower_bound
        b = other / self.upper_bound
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))


    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        if isinstance(other, Interval):
            a = self.lower_bound // other.lower_bound
            b = self.lower_bound // other.upper_bound
            c = self.upper_bound // other.lower_bound
            d = self.upper_bound // other.upper_bound
            return Interval(np.min([a, b, c, d], axis=0),
                            np.max([a, b, c, d], axis=0))
        a = self.lower_bound // other
        b = self.upper_bound // other
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __rfloordiv__(self, other):
        a = other // self.lower_bound
        b = other // self.upper_bound
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __mod__(self, other):
        if isinstance(other, Interval):
            a = self.lower_bound % other.lower_bound
            b = self.lower_bound % other.upper_bound
            c = self.upper_bound % other.lower_bound
            d = self.upper_bound % other.upper_bound
            return Interval(np.min([a, b, c, d], axis=0),
                            np.max([a, b, c, d], axis=0))
        a = self.lower_bound % other
        b = self.upper_bound % other
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __rmod__(self, other):
        a = other % self.lower_bound
        b = other % self.upper_bound
        return Interval(np.min([a, b], axis=0),
                        np.max([a, b], axis=0))

    def __neg__(self):
        return Interval(-self.upper_bound, -self.lower_bound)

    def __lshift__(self, s):
        return Interval(self.lower_bound << s, self.upper_bound << s)

    def __rshift__(self, s):
        return Interval(self.lower_bound >> s, self.upper_bound >> s)

    def astype(self, dtype):
        return Interval(dtype(self.lower_bound), dtype(self.upper_bound))

    def __nonzero__(self):
        return bool(np.all(self.lower_bound != self.upper_bound))

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, Interval):
            return bool(
                np.all(self.lower_bound == other.lower_bound) and
                np.all(self.upper_bound == other.upper_bound)
            )
        return bool(
            np.all(self.lower_bound == other) and
            np.all(self.upper_bound == other)
        )

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if isinstance(other, Interval):
            return bool(np.all(self.upper_bound < other.lower_bound))
        return bool(np.all(self.upper_bound < other))

    def __le__(self, other):
        if isinstance(other, Interval):
            return bool(np.all(self.upper_bound <= other.lower_bound))
        return bool(np.all(self.upper_bound <= other))

    def __gt__(self, other):
        if isinstance(other, Interval):
            return bool(np.all(self.lower_bound > other.upper_bound))
        return bool(np.all(self.lower_bound > other))

    def __ge__(self, other):
        if isinstance(other, Interval):
            return bool(np.all(self.lower_bound >= other.upper_bound))
        return bool(np.all(self.lower_bound >= other))

    def __getitem__(self, key):
        return Interval(self.lower_bound[key], self.upper_bound[key])

    def __contains__(self, other):
        if isinstance(other, Interval):
            return bool(np.all(self.lower_bound <= other.lower_bound) and
                        np.all(other.upper_bound <= self.upper_bound))
        return bool(np.all(self.lower_bound <= other) and
                    np.all(other <= self.upper_bound))

    def __hash_content__(self):
        lower_bound = self.lower_bound
        if hasattr(lower_bound, 'tobytes'):
            lower_bound = lower_bound.tobytes()
        upper_bound = self.upper_bound
        if hasattr(upper_bound, 'tobytes'):
            upper_bound = upper_bound.tobytes()
        return (lower_bound, upper_bound)

    def __hash__(self):
        return hash(self.__hash_content__())

interval = Interval

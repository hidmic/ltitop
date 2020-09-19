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

import numpy as np

from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit
from ltitop.arithmetic.fixed_point.representation import Representation

from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.helpers import type_uniform_binary_operator


@immutable_dataclass
class Number(Representation):

    @classmethod
    def from_value(cls, *args, **kwargs):
        unit = ProcessingUnit.active()
        return unit.represent(*args, rtype=cls, **kwargs)

    def __add__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.substract(self, other)

    def __rsub__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.substract(other, self)

    def __mul__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.multiply(self, other)

    __rmul__ = __mul__

    def __div__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.divide(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.divide(other, self)

    __rtruediv__ = __rdiv__

    def __mod__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return Number(unit.modulus(self, other))

    def __rmod__(self, other):
        if not isinstance(other, Representation):
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return unit.modulus(other, self)

    def __trunc__(self):
        unit = ProcessingUnit.active()
        return unit.truncate(self)

    def __ceil__(self):
        unit = ProcessingUnit.active()
        return unit.ceil(self)

    def __floor__(self):
        unit = ProcessingUnit.active()
        return unit.floor(self)

    def __round__(self):
        unit = ProcessingUnit.active()
        return unit.nearest(self)

    def __neg__(self):
        unit = ProcessingUnit.active()
        return unit.negate(self)

    def __eq__(self, other):
        if not self or not other:
            return not self and not other
        if not isinstance(other, Representation):
            if other not in self.format_.value_interval:
                return False
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return bool(np.all(unit.compare(self, other) == 0))

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if not isinstance(other, Representation):
            vi = self.format_.value_interval
            if np.any(other < vi.lower_bound):
                return False
            if np.any(other > vi.upper_bound):
                return True
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return bool(np.all(unit.compare(self, other) < 0))

    def __le__(self, other):
        if not isinstance(other, Representation):
            vi = self.format_.value_interval
            if np.any(other < vi.lower_bound):
                return False
            if np.any(other > vi.upper_bound):
                return True
            other = Number.from_value(other)
        unit = ProcessingUnit.active()
        return bool(np.all(unit.compare(self, other) <= 0))

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def __lshift__(self, n):
        unit = ProcessingUnit.active()
        return unit.lshift(self, n)

    def __rshift__(self, n):
        unit = ProcessingUnit.active()
        return unit.rshift(self, n)

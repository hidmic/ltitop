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

from ltitop.arithmetic.modular import wraparound
from ltitop.arithmetic.rounding import floor
from ltitop.common.annotation import annotated_function
from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.tracing import Traceable


class ProcessingUnit(metaclass=Traceable):
    @immutable_dataclass
    class Info:
        eps: mpmath.mpf
        min: mpmath.mpf
        max: mpmath.mpf

    __active = None

    @classmethod
    def active(cls):
        if ProcessingUnit.__active is None:
            raise RuntimeError("No active fixed point process unit")
        return ProcessingUnit.__active

    def __enter__(self):
        self.__last_active = ProcessingUnit.__active
        ProcessingUnit.__active = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ProcessingUnit.__active = self.__last_active

    def __init__(
        self,
        *,
        rounding_method=floor,
        overflow_behavior=wraparound,
        allows_overflow=True,
        allows_underflow=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        cls = type(self)
        self.__rounding_method = rounding_method
        self.__overflow_behavior = overflow_behavior

        if allows_overflow is True:
            allows_overflow = cls.__dict__.values()
        elif allows_overflow is False:
            allows_overflow = []

        if allows_underflow is True:
            allows_underflow = cls.__dict__.values()
        elif allows_underflow is False:
            allows_underflow = []

        self.represent = annotated_function(
            self.represent,
            allows_overflow=cls.represent in allows_overflow,
            allows_underflow=cls.represent in allows_underflow,
        )
        self.add = annotated_function(
            self.add,
            allows_overflow=cls.add in allows_overflow,
            allows_underflow=cls.add in allows_underflow,
        )
        self.substract = annotated_function(
            self.substract,
            allows_overflow=cls.substract in allows_overflow,
            allows_underflow=cls.substract in allows_underflow,
        )
        self.multiply = annotated_function(
            self.multiply,
            allows_overflow=cls.multiply in allows_overflow,
            allows_underflow=cls.multiply in allows_underflow,
        )
        self.divide = annotated_function(
            self.divide,
            allows_overflow=cls.divide in allows_overflow,
            allows_underflow=cls.divide in allows_underflow,
        )
        self.modulus = annotated_function(
            self.modulus,
            allows_overflow=cls.modulus in allows_overflow,
            allows_underflow=cls.modulus in allows_underflow,
        )
        self.lshift = annotated_function(
            self.lshift,
            allows_overflow=cls.rshift in allows_overflow,
        )
        self.negate = annotated_function(
            self.negate,
            allows_overflow=(
                cls.negate in allows_overflow
                or cls.substract in allows_overflow
                or cls.multiply in allows_overflow
            ),
        )
        self.compare = annotated_function(
            self.compare, allows_underflow=cls.compare in allows_underflow
        )

    @property
    def rounding_method(self):
        return self.__rounding_method

    @property
    def overflow_behavior(self):
        return self.__overflow_behavior

    def rinfo(self, **kwargs):
        raise NotImplementedError()

    def represent(self, value, **kwargs):
        raise NotImplementedError()

    def add(self, x, y):
        return NotImplemented

    def substract(self, x, y):
        return NotImplemented

    def multiply(self, x, y):
        return NotImplemented

    def divide(self, x, y):
        return NotImplemented

    def modulus(self, x, y):
        return NotImplemented

    def compare(self, x, y):
        return NotImplemented

    def truncate(self, x):
        return NotImplemented

    def floor(self, x):
        return NotImplemented

    def ceiling(self, x):
        return NotImplemented

    def negate(self, x):
        return NotImplemented

    def lshift(self, n):
        return NotImplemented

    def rshift(self, n):
        return NotImplemented

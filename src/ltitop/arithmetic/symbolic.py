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

import mpmath
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit, call_highest_priority
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import Function
from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S

from ltitop.arithmetic.floating_point import mpfloat


class BaseNumber(AtomicExpr):
    __slots__ = ()

    is_number = True
    # NOTE(hidmic): a hack to force higher precedence
    is_Rational = True  # sympy is a bit rough in here
    is_Number = True
    is_comparable = True

    _op_priority = Expr._op_priority

    @classmethod
    def _new(cls, num):
        return super().__new__(cls, num)

    def _eval_is_integer(self):
        return int(self._args[0]) == self._args[0]

    def _eval_is_negative(self):
        return self._args[0] < 0

    _eval_is_extended_negative = _eval_is_negative

    def _eval_is_positive(self):
        return self._args[0] > 0

    _eval_is_extended_positive = _eval_is_positive

    def _eval_is_zero(self):
        return self._args[0] == 0

    def __nonzero__(self):
        return self._args[0] != 0

    __bool__ = __nonzero__

    def floor(self):
        return self._new(math.floor(self._args[0]))

    def ceiling(self):
        return self._new(math.ceil(self._args[0]))

    def __float__(self):
        return float(self._args[0])

    def __int__(self):
        return int(self._args[0])

    __long__ = __int__

    def _as_mpf_val(self, prec):
        with mpmath.workprec(prec):
            return mpfloat(self._args[0])

    def __getattr__(self, name):
        return getattr(self._args[0], name)

    def __getitem__(self, key):
        return type(self)(self._args[0][key])

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__radd__")
    def __add__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.Zero:
                return self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] + other._args[0])
        return super().__add__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__add__")
    def __radd__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.Zero:
                return self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(other._args[0] + self._args[0])
        return super().__radd__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rsub__")
    def __sub__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.Zero:
                return self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] - other._args[0])
        return super().__sub__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__sub__")
    def __rsub__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.Zero:
                return self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(other._args[0] - self._args[0])
        return super().__rsub__(other)

    def __neg__(self):
        return self._new(-self._args[0])

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rmul__")
    def __mul__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.One:
                return self
            if other == -S.One:
                return -self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] * other._args[0])
        return super().__mul__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__mul__")
    def __rmul__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.One:
                return self
            if other == -S.One:
                return -self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] * other._args[0])
        return super().__mul__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rdiv__")
    def __div__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.One:
                return self
            if other == -S.One:
                return -self
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] / other._args[0])
        return super().__div__(other)

    __truediv__ = __div__

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__div__")
    def __rdiv__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(other._args[0] / self._args[0])
        return super().__rdiv__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__rmod__")
    def __mod__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if other == S.One:
                return type(self)(0)
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(self._args[0] % other._args[0])
        return super().__mod__(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__mod__")
    def __rmod__(self, other):
        if isinstance(other, (BaseNumber, Number)) and global_parameters.evaluate:
            if not isinstance(other, type(self)):
                other = type(self)(other)
            return self._new(other._args[0] % self._args[0])
        return super().__rmod__(other)

    def __hash__(self):
        return super().__hash__()

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__eq__")
    def __eq__(self, other):
        if not other.is_Number:
            return False
        if not other:
            return not self._args[0]
        if isinstance(other, type(self)):
            return self._args[0] == other._args[0]
        if isinstance(self._args[0], Basic):
            return self._args[0] == other
        return self._args[0] == mpfloat(other)

    def __ne__(self, other):
        return not (self == other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__lt__")
    def __lt__(self, other):
        if not other.is_Number:
            return super().__lt__(other)
        if not other:
            return self._args[0] < 0
        if isinstance(other, type(self)):
            return self._args[0] < other._args[0]
        if isinstance(self._args[0], Basic):
            return self._args[0] < other
        return self._args[0] < mpfloat(other)

    @_sympifyit("other", NotImplemented)
    @call_highest_priority("__le__")
    def __le__(self, other):
        if not other.is_Number:
            return super().__le__(other)
        if not other:
            return self._args[0] <= 0
        if isinstance(other, type(self)):
            return self._args[0] <= other._args[0]
        if isinstance(self._args[0], Basic):
            return self._args[0] <= other
        return self._args[0] <= mpfloat(other)

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def __lshift__(self, n):
        return self._new(self._args[0] << n)

    def __rshift__(self, n):
        return self._new(self._args[0] >> n)


class LoadExponent(Function):
    @classmethod
    def eval(cls, x, k):
        if isinstance(x, (BaseNumber, Number)) and isinstance(k, (BaseNumber, Number)):
            try:
                return x << k if k > 0 else x >> -k
            except TypeError:
                return x * 2 ** k

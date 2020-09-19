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

import random
import sympy

from sympy.core.sympify import _sympify as _sympify_
from sympy.core.parameters import global_parameters

import ltitop.algorithms.expressions as expressions
from ltitop.arithmetic.symbolic import ldexp


class NonAssociativeOp(sympy.Basic):

    __slots__ = ('_expr')

    def __new__(cls, *args, evaluate=None, _sympify=True):
        if _sympify:
            args = list(map(_sympify_, args))
        if evaluate is None:
            evaluate = global_parameters.evaluate
        return cls._from_args(args, evaluate)

    @classmethod
    def _from_args(cls, args, evaluate):
        if not args:
            return cls.basefunc.identity
        if len(args) == 1:
            return args[0]
        *head, tail = args
        expr = cls.basefunc(
            cls._from_args(head, evaluate=evaluate), tail,
            evaluate=evaluate, _sympify=False
        )
        if expr.func is not cls.basefunc:
            return expr
        assert len(expr.args) == 2
        obj = super().__new__(cls, *expr.args)
        obj._expr = expr
        return obj

    def _anycode(self, printer):
        return '(' + printer._print(self._expr)  + ')'

    def __getattr__(self, name):
        if name.startswith('_') and name.endswith('code'):
            return self._anycode
        raise super().__getattr__(name)

    def _eval_subs(self, old, new):
        return self._expr._eval_subs(old, new)

    def _eval_evalf(self, prec):
        return self._expr._eval_evalf(prec)


class NonAssociativeAdd(sympy.Expr, NonAssociativeOp):

    __slots__ = ()

    basefunc = sympy.Add


class NonAssociativeMul(sympy.Expr, NonAssociativeOp):

    __slots__ = ()

    basefunc = sympy.Mul


def rotate_left(expr):
    pivot = expr
    rotator = pivot.args[1]
    if pivot.func is not rotator.func:
        raise ValueError(f'Cannot rotate left {expr}')
    return rotator.func(
        pivot.func(
            pivot.args[0], rotator.args[0]
        ), rotator.args[1]
    )


def can_rotate_left(expr):
    pivot = expr
    rotator = pivot.args[1]
    return pivot.func is rotator.func


def rotate_right(expr):
    pivot = expr
    rotator = pivot.args[0]
    if pivot.func is not rotator.func:
        raise ValueError(f'Cannot rotate right {expr}')
    return rotator.func(
        rotator.args[0], pivot.func(
            rotator.args[1], pivot.args[1]
        )
    )


def can_rotate_right(expr):
    pivot = expr
    rotator = pivot.args[0]
    return pivot.func is rotator.func


@expressions.modifier
def associative(expr):
    def predicate(expr):
        if issubclass(expr.func, NonAssociativeOp):
            expr = expr.func.basefunc(*expr.args)
        return expr, False
    return expressions.reconfigure(expr, predicate)


def nonassociative(variant):
    @expressions.modifier
    def implementation(expr):
        r = random.Random(variant)
        def should_rotate_left(expr):
            return can_rotate_left(expr) and bool(r.getrandbits(1))
        def should_rotate_right(expr):
            return can_rotate_right(expr) and bool(r.getrandbits(1))

        mapping = {cls.basefunc: cls for cls in NonAssociativeOp.__subclasses__()}
        ignored = {sympy.Indexed}

        def predicate(expr):
            if expr.func in ignored:
                return expr, True
            if expr.func in mapping:
                func = mapping[expr.func]
                expr = func(*expr.args)
            if issubclass(expr.func, NonAssociativeOp):
                while should_rotate_left(expr) and can_rotate_left(expr):
                    expr = rotate_left(expr)
                while should_rotate_right(expr) and can_rotate_right(expr):
                    expr = rotate_right(expr)
            return expr, False
        return expressions.reconfigure(expr, predicate)
    return implementation

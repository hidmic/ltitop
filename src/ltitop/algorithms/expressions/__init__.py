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

import dataclasses
import functools

from ltitop.algorithms import Algorithm
from ltitop.algorithms.statements import Assignment


def reconfigure(expr, predicate):
    expr, done = predicate(expr)
    if done or not expr.args:
        return expr
    return expr.func(*(reconfigure(arg, predicate) for arg in expr.args))


def modifier(func):
    @functools.wraps(func)
    def _modifier(algorithm):
        if not isinstance(algorithm, Algorithm):
            return func(algorithm)
        modified_procedure = []
        for statement in algorithm.procedure:
            if isinstance(statement, Assignment):
                rhs = statement.rhs
                if isinstance(rhs, tuple):
                    rhs = tuple(func(expr) for expr in rhs)
                else:
                    rhs = func(rhs)
                statement = Assignment(statement.lhs, rhs)
            modified_procedure.append(statement)
        return dataclasses.replace(algorithm, procedure=tuple(modified_procedure))

    return _modifier

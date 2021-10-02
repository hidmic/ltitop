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
from typing import Any, Dict, Iterable, Tuple, Union

import sympy

from ltitop.algorithms.statements import Statement
from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.helpers import astuple

# TODO(hidmic): replace all of this with sympy.codegen AST


def subs(
    expressions: Union[sympy.Expr, Iterable[sympy.Expr]],
    values: Union[Any, Iterable[Any]],
) -> Dict[sympy.Expr, sympy.Basic]:
    expressions = astuple(expressions)
    values = astuple(values)
    if len(expressions) != len(values):
        raise ValueError(f"Mismatch between {expressions} and {values}")
    return {e: sympy.sympify(v) for e, v in zip(expressions, values)}


@immutable_dataclass
class Algorithm:
    procedure: Tuple[Statement, ...]
    inputs: Tuple[sympy.Expr, ...] = ()
    states: Tuple[sympy.Expr, ...] = ()
    outputs: Tuple[sympy.Expr, ...] = ()
    parameters: Tuple[sympy.Expr, ...] = ()

    def __post_init__(self):
        for f in dataclasses.fields(self):
            object.__setattr__(self, f.name, astuple(getattr(self, f.name)))

    def apply(self, modifiers):
        outcome = self
        for modifier in modifiers:
            outcome = modifier(outcome)
        return outcome

    def to_function(self, parameters):
        steps = []
        variables = set()
        arguments = list(self.inputs) + list(self.states)
        constants = dict(zip(self.parameters, parameters))
        for statement in self.procedure:
            change = statement.perform()
            for var, expression in change.items():
                local_variables = [v for v in variables if expression.has(v)]
                local_constants = [c for c in constants if expression.has(c)]
                import ltitop.algorithms.expressions.arithmetic as arithmetic
                import ltitop.arithmetic.symbolic as symbolic

                func = functools.partial(
                    sympy.lambdify(
                        local_constants + arguments + local_variables,
                        expression,
                        modules=[symbolic, arithmetic, "numpy"],
                    ),
                    *[constants[c] for c in local_constants],
                )
                steps.append((var, func, local_variables))
                if var not in self.states:
                    variables.add(var)

        def _function(inputs, states):
            scope = {}
            for var, func, args in steps:
                scope[var] = func(*inputs, *states, *(scope[a] for a in args))
            return ([scope[s] for s in self.states], [scope[o] for o in self.outputs])

        return _function

    def define(self, inputs=None, states=None, parameters=None):
        scope = {}
        if inputs is not None:
            scope.update(subs(self.inputs, inputs))
        if states is not None:
            scope.update(subs(self.states, states))
        if parameters is not None:
            scope.update(subs(self.parameters, parameters))
        if not scope:
            raise ValueError("Scope is empty")
        return scope

    def run(self, scope):
        scope = dict(scope)
        for change in self.step(scope):
            scope.update(change)
        return scope

    def step(self, scope):
        for statement in self.procedure:
            yield statement.perform(scope)

    def collect(self, scope):
        return (
            tuple(state.subs(scope.items()) for state in self.states),
            tuple(output.subs(scope.items()) for output in self.outputs),
        )

    def __str__(self):
        return "\n".join(
            [
                "Algorithm:",
                "  Inputs: {}".format(", ".join(map(str, self.inputs))),
                "  States: {}".format(", ".join(map(str, self.states))),
                "  Outputs: {}".format(", ".join(map(str, self.outputs))),
                "  Parameters: {}".format(", ".join(map(str, self.parameters))),
                "",
            ]
            + [f"  {statement}" for statement in self.procedure]
        )

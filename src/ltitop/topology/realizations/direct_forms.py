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
import scipy.signal as signal
import sympy
import sys

from ltitop.algorithms import Algorithm
from ltitop.algorithms.statements import Assignment
import ltitop.algorithms.expressions.arithmetic as arithmetic
from ltitop.arithmetic.errors import UnderflowError
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.symbolic import ldexp
import ltitop.arithmetic.limits as limits
from ltitop.common.arrays import asvector_if_possible
from ltitop.common.arrays import within
from ltitop.common.helpers import identity
from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.memoization import memoize
from ltitop.models.analysis import output_range
from ltitop.topology.realizations import Realization


@immutable_dataclass
class DirectForm(Realization):

    @immutable_dataclass(iterable=True)
    class Parameters:
        b: np.ndarray
        a: np.ndarray

        def __post_init__(self):
            if len(self.b) == 0:
                raise ValueError('no b[n] coefficients')
            if len(self.a) == 0:
                raise ValueError('no a[n] coefficients')
            if self.a[0] != 1:
                raise ValueError('a[0] is not normalized to 1')

        def __eq__(self, other):
            return np.array_equal(self.b, other.b) and \
                np.array_equal(self.a, other.a)

        def __hash__(self):
            return hash(tuple(self.b) + tuple(self.a))

    @immutable_dataclass(iterable=True)
    class ScaledParameters(Parameters):
        k: int

        def __eq__(self, other):
            return self.k == other.k and \
                np.array_equal(self.b, other.b) and \
                np.array_equal(self.a, other.a)

        def __hash__(self):
            return hash(tuple(self.b) + tuple(self.a) + (self.k,))

    @classmethod
    def _make_direct_algorithm(cls, *, n_b, n_a, has_k):
        raise NotImplementedError()

    @classmethod
    def _make_algorithm(cls, *, n_b, n_a, has_k):
        if n_b == 0:
            raise ValueError(f'no b[n] coefficients')
        if n_a == 0:
            raise ValueError(f'no a[n] coefficients')
        return cls._make_direct_algorithm(
            n_b=n_b, n_a=n_a, has_k=has_k
        )

    __slots__ = ('_variant')

    def __init__(self, parameters, variant=None):
        modifiers = []
        if variant is not None:
            modifiers.append(arithmetic.nonassociative(variant))
        object.__setattr__(self, '_variant', variant)
        algorithm = self.make_algorithm(
            n_b=len(parameters.b),
            n_a=len(parameters.a),
            has_k=hasattr(parameters, 'k'),
            modifiers=tuple(modifiers)
        )
        object.__setattr__(self, 'parameters', parameters)
        object.__setattr__(self, 'algorithm', algorithm)

    def __repr__(self):
        return '{}(parameters={!r}, variant={!r})'.format(
            type(self).__name__, self.parameters, self._variant
        )

    def process(self, inputs, initial_states=None):
        inputs = np.c_[inputs]
        if initial_states is None:
            initial_states = [np.zeros(shape) for shape in self.states]
        states, outputs = super().process(inputs, initial_states)
        states = np.asarray(states)
        outputs = np.asarray(outputs)
        return states, outputs

    def to_function(self, initial_state=None):
        return super().to_function(initial_state)

    @classmethod
    def from_model(cls, model, dtype=None, **kwargs):
        if not isinstance(model, signal.dlti):
            model = signal.dlti(*model)
        model = model.to_tf()
        if len(model.num.shape) > 1:
            raise NotImplementedError('MISO and MIMO systems not supported')
        num = model.num
        den = model.den
        if len(num) > len(den):
            raise ValueError('Non causal model')
        # From positive to negative powers
        if np.any(num):
            pad_width = len(den) - len(num)
            num = np.pad(num, (pad_width, 0))
            b = np.trim_zeros(num, 'b')
        else:
            b = num[0:1]
        a = np.trim_zeros(den, 'b')
        # Normalize leading denominator coefficient
        b = b / a[0]
        a = a / a[0]
        k = None  # may have to scale coefficients
        if dtype is not None:
            coefficients = np.concatenate([b, a[1:]])
            mask = ~within(coefficients, limits.interval(dtype))
            if np.any(mask):
                # Scale coefficients
                k = max(int(math.ceil(math.log2(
                    np.max(np.abs(coefficients[mask]))
                ))), 1)
                coefficients = np.ldexp(coefficients, -k)
            coefficients = np.array([
                dtype(coeff) for coeff in coefficients
            ])
            b = np.trim_zeros(coefficients[:len(b)], 'b')
            if b.size == 0:
                b = coefficients[0:1]
                a = a[0:1]
                k = None
            else:
                a = np.concatenate((
                    a[0:1], np.trim_zeros(coefficients[len(b):], 'b')
                ))
        if k is not None:
            parameters = DirectForm.ScaledParameters(b, a, k)
        else:
            parameters = DirectForm.Parameters(b, a)
        return cls(parameters, **kwargs)

    def _make_model(self):
        b = self.parameters.b.astype(float)
        a = self.parameters.a.astype(float)
        if not self.states:
            # Handle degenerate model
            return signal.dlti(b[0], a[0])
        # From negative to positive powers
        pad_width = len(a) - len(b)
        if pad_width > 0:
            b = np.pad(b, (0, pad_width))
        elif pad_width < 0:
            a = np.pad(a, (0, -pad_width))
        num = np.trim_zeros(b, 'f')
        if num.size == 0:
            num = b[0:1]
        den = np.trim_zeros(a, 'f')
        if hasattr(self.parameters, 'k'):
            num = np.ldexp(num, self.parameters.k)
            den[1:] = np.ldexp(den[1:], self.parameters.k)
        return signal.dlti(num, den)

    def _make_stateful_computation_error_model(self):
        raise NotImplementedError()

    def _make_computation_error_model(self):
        if not self.states:
            b = int(bool(self.parameters.b[0]))
            return signal.dlti(b, 1.)
        return self._make_stateful_computation_error_model()

    def computation_error_bounds(self, input_range, state_ranges=None):
        if state_ranges is None and self.states:
            state_ranges = []
            for model in self.state_observer_models:
                state_range = output_range(model, mpfloat(input_range))
                state_range = interval(
                    lower_bound=asvector_if_possible(state_range.lower_bound),
                    upper_bound=asvector_if_possible(state_range.upper_bound),
                )
                state_ranges.append(state_range)
        return super().computation_error_bounds([input_range], state_ranges)

    def _pretty(self, printer):
        z = sympy.UnevaluatedExpr(sympy.Symbol('z'))
        k = getattr(self.parameters, 'k', 0)
        num = None
        for i, c in enumerate(self.parameters.b):
            if c:
                term = printer._print(
                    np.ldexp(float(c), k) * z**-i)
                num = num + term if num else term
        if num is None:
            num = 0
        den = printer._print(float(self.parameters.a[0]))
        for i, c in enumerate(self.parameters.a[1:], start=1):
            if c:
                den += printer._print(np.ldexp(float(c), k) * z**-i)
        return printer._print(num) / printer._print(den)


@immutable_dataclass(init=False, repr=False)
class DirectFormI(DirectForm):

    @classmethod
    def _make_direct_algorithm(cls, *, n_b, n_a, has_k):
        b = sympy.IndexedBase('b', shape=(n_b,))
        a = sympy.IndexedBase('a', shape=(n_a,))
        x = sympy.IndexedBase('x')
        y = sympy.IndexedBase('y')
        i, n = sympy.symbols('i n')

        inputs = x[n]
        outputs = y[n]
        if has_k:
            k = sympy.symbols('k')
            parameters = b, a, k
            f = lambda _: ldexp(_, k)
        else:
            parameters = b, a
            f = identity
        states = [x[n - j] for j in range(1, n_b)]
        states.extend(y[n - j] for j in range(1, n_a))

        procedure = [Assignment(lhs=y[n], rhs=f(
            sympy.Sum(b[i] * x[n - i], (i, 0, n_b - 1)) -
            sympy.Sum(a[i] * y[n - i], (i, 1, n_a - 1))
        ).doit())]

        if n_b > 1:
            procedure.append(Assignment(
                lhs=tuple(x[n - j] for j in reversed(range(1, n_b))),
                rhs=tuple(x[n - j] for j in reversed(range(n_b - 1)))
            ))
        if n_a > 1:
            procedure.append(Assignment(
                lhs=tuple(y[n - j] for j in reversed(range(1, n_a))),
                rhs=tuple(y[n - j] for j in reversed(range(n_a - 1)))
            ))

        return Algorithm(
            inputs=inputs, outputs=outputs, states=states,
            parameters=parameters, procedure=procedure
        )

    def _make_stateful_computation_error_model(self):
        den = self.parameters.a.astype(float)
        if hasattr(self.parameters, 'k'):
            den[1:] = np.ldexp(den[1:], self.parameters.k)
        num = np.zeros_like(den); num[0] = 1.
        return signal.dlti(num, den)

    def _make_state_observer_models(self):
        observers = []
        n_b = len(self.parameters.b)
        if n_b > 1:
            den = np.zeros(n_b); den[0] = 1.
            for i in range(1, n_b):
                obs = signal.dlti(1., den[:i + 1])
                observers.append(obs)
        n_a = len(self.parameters.a)
        if n_a > 1:
            model = self.model
            num = model.num
            n_den = len(model.den)
            den = np.zeros(n_den + n_a - 1)
            den[:n_den] = model.den
            for i in range(1, n_a):
                obs = signal.dlti(num, den[:n_den + i])
                observers.append(obs)
        return tuple(observers)


@immutable_dataclass(init=False, repr=False)
class DirectFormII(DirectForm):

    __init__ = DirectForm.__init__

    @classmethod
    def _make_direct_algorithm(cls, *, n_b, n_a, has_k):
        n_s = max(n_a, n_b) - 1

        b = sympy.IndexedBase('b', shape=(n_b,))
        a = sympy.IndexedBase('a', shape=(n_a,))
        x = sympy.IndexedBase('x')
        y = sympy.IndexedBase('y')
        i, n = sympy.symbols('i n')

        inputs = x[n]
        outputs = y[n]

        if has_k:
            k = sympy.symbols('k')
            parameters = b, a, k
            f = lambda _: ldexp(_, k)
        else:
            parameters = b, a
            f = identity

        if n_s > 0:
            s = sympy.IndexedBase('s')
            sv = sympy.IndexedBase('𝐬', shape=(n_s,))
            procedure = [
                Assignment(lhs=tuple(s[n - j] for j in range(1, n_s + 1)), rhs=sv),
                Assignment(lhs=s[n], rhs=(x[n] - f(
                    sympy.Sum(a[i] * s[n - i], (i, 1, n_a - 1)))
                ).doit()),
                Assignment(lhs=y[n], rhs=f(
                    sympy.Sum(b[i] * s[n - i], (i, 0, n_b - 1))
                ).doit()),
                Assignment(lhs=sv, rhs=tuple(s[n - j] for j in range(n_s)))
            ]
            states = (sv,)
        else:
            procedure = Assignment(lhs=y[n], rhs=f(b[0] * x[n]))
            states = ()

        return Algorithm(
            inputs=inputs, outputs=outputs, states=states,
            parameters=parameters, procedure=procedure
        )

    def _make_stateful_computation_error_model(self):
        b = self.parameters.b.astype(float)
        a = self.parameters.a.astype(float)
        if hasattr(self.parameters, 'k'):
            b = np.ldexp(b, self.parameters.k)
            a[1:] = np.ldexp(a[1:], self.parameters.k)

        n_s = self.states[0][0]
        b = np.pad(b, (0, n_s + 1 - len(b)))
        a = np.pad(a, (0, n_s + 1 - len(a)))

        if n_s > 1:
            A = np.zeros((n_s, n_s))
            A[0, :] = -a[1:n_s + 1]
            A[1:, :-1] = np.eye(n_s - 1)
            C = np.array([
                -b[0] * a[i] + b[i]
                for i in range(1, n_s + 1)
            ])
        else:
            A = -a[1]
            C = -b[0] * a[1] + b[1]

        B = np.zeros((n_s, 2))
        B[0, 0] = 1
        D = np.array([b[0], 1])

        return signal.dlti(A, B, C, D)

    def _make_state_observer_models(self):
        a = self.parameters.a.astype(float)
        if hasattr(self.parameters, 'k'):
            a[1:] = np.ldexp(a[1:], self.parameters.k)
        n_s = self.states[0][0]
        A = np.zeros((n_s, n_s))
        A[0, :len(a) - 1] = -a[1:]
        A[1:, :-1] = np.eye(n_s - 1)
        B = np.zeros((n_s, 1))
        B[0, 0] = 1
        C = np.eye(n_s)
        D = np.zeros((n_s, 1))
        return signal.dlti(A, B, C, D),


@immutable_dataclass(init=False, repr=False)
class TransposedDirectFormII(DirectForm):

    __init__ = DirectForm.__init__

    @classmethod
    def _make_direct_algorithm(cls, *, n_b, n_a, has_k):
        b = sympy.IndexedBase('b', shape=(n_b,))
        a = sympy.IndexedBase('a', shape=(n_a,))
        x = sympy.IndexedBase('x')
        y = sympy.IndexedBase('y')
        n = sympy.Symbol('n')

        inputs = x[n]
        outputs = y[n]

        if has_k:
            k = sympy.Symbol('k')
            parameters = b, a, k
            f = lambda _: ldexp(_, k)
        else:
            parameters = b, a
            f = identity

        n_s = max(n_a, n_b) - 1
        if n_s > 0:
            sv = sympy.IndexedBase('𝐬', shape=(n_s,))
            s = [sympy.IndexedBase(f's{i}') for i in range(n_s)]

            b = sympy.Array([b[i] if i < n_b else 0 for i in range(n_s + 1)])
            a = sympy.Array([a[i] if i < n_a else 0 for i in range(n_s + 1)])

            procedure = [Assignment(
                lhs=tuple(s[i][n] for i in range(n_s)), rhs=sv
            )]
            procedure.append(Assignment(
                lhs=y[n], rhs=f(b[0] * x[n] + s[0][n])
            ))
            procedure.extend(Assignment(
                lhs=s[i][n - 1], rhs=(
                    b[i + 1] * x[n] - a[i + 1] * y[n] + s[i + 1][n]
                )
            ) for i in range(n_s - 1))
            procedure.append(Assignment(
                lhs=s[n_s - 1][n - 1], rhs=(
                    b[n_s] * x[n] - a[n_s] * y[n]
                )
            ))
            procedure.append(Assignment(
                lhs=sv, rhs=tuple(s[i][n - 1] for i in range(n_s))
            ))
            states = (sv,)
        else:
            procedure = Assignment(lhs=y[n], rhs=f(b[0] * x[n]))
            states = ()

        return Algorithm(
            inputs=inputs, outputs=outputs, states=states,
            parameters=parameters, procedure=procedure
        )

    def _make_stateful_computation_error_model(self):
        a = self.parameters.a.astype(float)
        if hasattr(self.parameters, 'k'):
            a[1:] = np.ldexp(a[1:], self.parameters.k)
        n_s = self.states[0][0]
        a = np.pad(a, (0, n_s + 1 - len(a)))

        if n_s > 1:
            A = np.zeros((n_s, n_s))
            A[:, 0] = -a[1:n_s + 1]
            A[:-1, 1:] = np.eye(n_s - 1)

            C = np.zeros((1, n_s))
            C[0, 0] = 1
        else:
            A = -a[1]
            C = 1

        B = np.zeros((n_s, n_s + 1))
        B[:, 0] = -a[1:n_s + 1]
        B[:, 1:] = np.eye(n_s)

        D = np.zeros((1, n_s + 1))
        D[0, 0] = 1

        return signal.dlti(A, B, C, D)

    def _make_state_observer_models(self):
        b = self.parameters.b.astype(float)
        a = self.parameters.a.astype(float)
        if hasattr(self.parameters, 'k'):
            b = np.ldexp(b, self.parameters.k)
            a[1:] = np.ldexp(a[1:], self.parameters.k)
        n_s = self.states[0][0]
        a = np.pad(a, (0, n_s + 1 - len(a)))
        b = np.pad(b, (0, n_s + 1 - len(b)))

        A = np.zeros((n_s, n_s))
        A[:, 0] = -a[1:n_s + 1]
        A[:-1, 1:] = np.eye(n_s - 1)
        B = np.c_[b[1:n_s + 1] - a[1:n_s + 1] * b[0]]
        C = np.eye(n_s)
        D = np.zeros((n_s, 1))
        return signal.dlti(A, B, C, D),

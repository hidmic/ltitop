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

import typing

import numpy as np
import scipy.signal as signal

from ltitop.algorithms import Algorithm
from ltitop.arithmetic.error_bounded import error_bounded
from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.interval import interval
from ltitop.common.arrays import empty_of
from ltitop.common.dataclasses import Dataclass
from ltitop.common.dataclasses import immutable_dataclass
from ltitop.common.memoization import memoize
from ltitop.models.analysis import output_range


@immutable_dataclass
class Realization:
    parameters: Dataclass
    algorithm: Algorithm

    @classmethod
    def _make_algorithm(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    @memoize
    def make_algorithm(cls, *, modifiers=None, **kwargs):
        algorithm = cls._make_algorithm(**kwargs)
        if modifiers:
            algorithm = algorithm.apply(modifiers)
        return algorithm

    @property
    @memoize
    def update_function(self):
        return self.algorithm.to_function(tuple(self.parameters))

    def process(self, inputs, initial_state):
        states = [None] * (len(inputs) + 1)
        states[0] = initial_state
        outputs = [None] * len(inputs)
        update = self.update_function
        for k in range(len(inputs)):
            states[k + 1], outputs[k] = \
                update(inputs[k], states[k])
        return states, outputs

    def to_function(self, initial_state):
        return (lambda inputs: self.process(inputs, initial_state)[1])

    @property
    @memoize
    def inputs(self):
        inputs = []
        for input_ in self.algorithm.inputs:
            try:
                shape = tuple(int(n) for n in input_.shape)
            except:
                shape = ()
            inputs.append(shape)
        return tuple(inputs)

    @property
    @memoize
    def states(self):
        states = []
        for state in self.algorithm.states:
            try:
                shape = tuple(int(n) for n in state.shape)
            except:
                shape = ()
            states.append(shape)
        return tuple(states)

    @property
    @memoize
    def outputs(self):
        outputs = []
        for output in self.algorithm.outputs:
            try:
                shape = tuple(int(n) for n in output.shape)
            except:
                shape = ()
            outputs.append(shape)
        return tuple(outputs)

    @classmethod
    def from_model(cls, model, **kwargs):
        raise NotImplementedError()

    def _make_model(self):
        raise NotImplementedError()

    @property
    @memoize
    def model(self):
        return self._make_model()

    def _make_state_observer_models(self):
        raise NotImplementedError()

    @property
    @memoize
    def state_observer_models(self):
        if len(self.states) == 0:
            raise RuntimeError('No states to observe')
        return self._make_state_observer_models()

    def _make_computation_error_model(self):
        raise NotImplementedError()

    @property
    @memoize
    def computation_error_model(self):
        return self._make_computation_error_model()

    def computation_error_bounds(self, input_ranges, state_ranges):
        state_ranges = state_ranges or []
        domain = self.algorithm.define(
            inputs=tuple(error_bounded(fixed(r)) for r in input_ranges),
            states=tuple(error_bounded(fixed(r)) for r in state_ranges),
            parameters=tuple(self.parameters)
        )
        error_bounds = []
        unit = ProcessingUnit.active()
        with unit.trace() as trace:
            scope = dict(domain)
            compare = type(unit).compare
            for change in self.algorithm.step(scope):
                if len(trace) > 0:
                    if any(func is not compare for func, *_ in trace):
                        for value in change.values():
                            if isinstance(value, tuple):
                                error_bounds.extend(
                                    elem.error_bounds for elem in value)
                            else:
                                error_bounds.append(value.error_bounds)
                    trace.clear()
                scope.update(change)
                scope.update(domain)
        if not error_bounds:
            return interval(0)
        if len(error_bounds) == 1:
            return error_bounds[0]
        return interval(
            lower_bound=np.c_[[eb.lower_bound for eb in error_bounds]],
            upper_bound=np.c_[[eb.upper_bound for eb in error_bounds]]
        )

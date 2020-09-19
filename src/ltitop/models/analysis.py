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

import contextlib
import itertools
import functools
import math
import os
import warnings

import numpy as np
import scipy.signal as signal
import scipy.signal.ltisys as sys


def WCPG_ABCD(A, B, C, D, rel_tol=1e-6, nmax=10000):
    if not np.any(A @ A):
        # Only poles at zero -> FIR filter
        return (
            np.absolute(D) +
            np.absolute(C @ B) +
            np.absolute(C @ A @ B)
        )  # series expansion
    if np.any(np.abs(np.linalg.eigvals(A)) >= 1):
        raise ValueError(
            f'System matrix {A} has eigenvalues'
            ' larger than 1, output diverges')
    # TODO(hidmic): do better than brute force
    WCPG = D
    Ap = np.eye(A.shape[0])
    WCPG_terms = np.zeros_like(D)
    for n in range(math.ceil(nmax / 100)):
        WCPG_terms[:] = 0.
        for _ in range(n * 100, (n + 1) * 100):
            WCPG_terms += np.abs(C @ Ap @ B)
            Ap = Ap @ A
        WCPG += WCPG_terms
        if np.all((WCPG_terms / WCPG) <= rel_tol):
            break  # early
    else:
        warnings.warn(
            (f'Could not achieve required tolerance ({rel_tol}) '
             'in worst case peak gain computation after summing '
             f'~{nmax * 100} terms'), RuntimeWarning)
    return WCPG

import ltitop.algebra.polynomials as poly
from ltitop.arithmetic.interval import Interval
from ltitop.common.arrays import asscalar_if_possible


@functools.lru_cache(maxsize=256)
def is_stable(model, tol=1e-16):
    if not isinstance(model, sys.dlti):
        model = sys.dlti(*model)
    return np.all(np.absolute(model.poles) < (1. - tol))

@functools.lru_cache(maxsize=256)
def spectral_radius(model):
    if not isinstance(model, sys.dlti):
        model = sys.dlti(*model)
    poles = model.poles
    if poles.size == 0:
        return None
    return np.max(np.absolute(poles))

@functools.lru_cache(maxsize=256)
def dc_gain(model):
    if not isinstance(model, sys.dlti):
        model = sys.dlti(*model)
    if isinstance(model, sys.ZerosPolesGainDiscrete):
        model = model.to_tf()
    if isinstance(model, sys.TransferFunctionDiscrete):
        if model.outputs != 1:
            raise NotImplementedError('SISO transfer functions only')
        return np.sum(model.num) / np.sum(model.den)
    if isinstance(model, sys.StateSpaceDiscrete):
        return model.C @ np.linalg.inv(
            np.eye(*model.A.shape) - model.A
        ) @ model.B + model.D
    raise TypeError(f'Cannot compute DC gain of {model}')


@functools.lru_cache(maxsize=256)
def worst_case_peak_gain(model):
    if not isinstance(model, sys.dlti):
        model = sys.dlti(*model)
    if isinstance(model, sys.StateSpaceDiscrete):
        return WCPG_ABCD(model.A.astype(np.float64),
                         model.B.astype(np.float64),
                         model.C.astype(np.float64),
                         model.D.astype(np.float64))
    if isinstance(model, sys.TransferFunctionDiscrete):
        if model.outputs != 1:
            raise NotImplementedError('SISO transfer functions only')
        return worst_case_peak_gain(model.to_ss())[0, 0]
    raise TypeError(f'Cannot compute worst case peak gain of {model}')

wcpg = worst_case_peak_gain


@functools.lru_cache(maxsize=128)
def output_range(model, input_range):
    if not isinstance(model, sys.dlti):
        model = sys.dlti(*model)

    if not isinstance(input_range, Interval):
        input_range = Interval(input_range)

    mean_input = (
        input_range.upper_bound +
        input_range.lower_bound
    ) / 2
    input_delta = (
        input_range.upper_bound -
        input_range.lower_bound
    ) / 2

    if model.inputs > 1:
        mean_output = dc_gain(model) @ mean_input
        output_delta = worst_case_peak_gain(model) @ input_delta
    else:
        mean_output = dc_gain(model) * mean_input
        output_delta = worst_case_peak_gain(model) * input_delta

    if model.outputs == 1:
        mean_output = asscalar_if_possible(mean_output)
        output_delta = asscalar_if_possible(output_delta)

    a = mean_output - output_delta; b = mean_output + output_delta

    return Interval(np.min([a, b], axis=0), np.max([a, b], axis=0))

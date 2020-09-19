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

import functools

import numpy as np
import scipy.signal as signal

import ltitop.algebra.polynomials as poly
import ltitop.algebra.rational_functions as rf


def model_decomposition(operator):
    @functools.wraps(operator)
    def __wrapper(model, *args, **kwargs):
        model = model._as_zpk()
        decomposition = operator(
            model.zeros, model.poles, model.gain,
            *args, dt=model.dt, **kwargs
        )
        kwargs = {}
        if model.dt is not None:
            cls = signal.ltisys.ZerosPolesGainDiscrete
            kwargs['dt'] = model.dt
        else:
            cls = signal.ltisys.ZerosPolesGainContinuous
        return [cls(*args, **kwargs) for args in decomposition]
    return __wrapper

def model_composition(operator):
    @functools.wraps(operator)
    def __wrapper(models):
        dt = models[0].dt
        if not all(model.dt == dt for model in models[1:]):
            raise ValueError(
                'Cannot mix continuous models with discrete models nor'
                ' discrete models with different sampling frequencies')
        models = [model._as_zpk() for model in models]
        args = operator([(model.zeros, model.poles, model.gain) for model in models])
        kwargs = {}
        if dt is not None:
            cls = signal.ltisys.ZerosPolesGainDiscrete
            kwargs['dt'] = dt
        else:
            cls = signal.ltisys.ZerosPolesGainContinuous
        return cls(*args, **kwargs)
    return __wrapper

@model_composition
def series_composition(zpks):
    # TODO(hidmic): support MIMO systems
    return rf.product(zpks)

@model_composition
def parallel_composition(zpks):
    # TODO(hidmic): support MIMO systems
    return rf.summation(zpks)

def _simplify_continuous_time_model(z, p, k, tol=1e-16):
    if len(z) == 0:
        return z, p, k
    # TODO(hidmic): do better than this
    p_max = p[np.argmax(np.abs(p))]
    wo = np.abs(np.imag(p_max))
    wo += np.abs(np.real(p_max))
    zo = 1j*wo
    ez = np.array([zo, -zo])
    z = np.ma.array(z, mask=np.zeros_like(z))
    for i in range(len(z)):
        ezi = ez
        zi = z[i]
        wi = np.imag(zi)
        if np.abs(wi) <= wo:
            ezi = np.array([1j*wi, -zo * np.copysign(1, wi)])
        # NOTE(hidmic): what about phase changes?
        z.mask[i] = True
        zm = z if not np.all(z.mask) else np.array([])
        error = ((np.abs(ezi - zi) - np.abs(zi)) *
                 np.abs(rf.evaluate(zm, p, k, ezi)))
        if np.all(error < tol):
            k *= np.abs(zi)
        else:
            z.mask[i] = False
    return z.compressed(), p, k

def _simplify_discrete_time_model(z, p, k, tol=1e-16):
    if len(z) == 0:
        return z, p, k
    # TODO(hidmic): do better than this
    mask = np.abs(z) < 10. * np.max(np.abs(p))
    k *= np.prod(np.abs(z[~mask]))
    z = z[mask]
    return z, p, k

@model_decomposition
def parallel_decomposition(*args, dt, tol=1e-16, **kwargs):
    # TODO(hidmic): support MIMO systems
    if dt is None:
        simplify_model = _simplify_continuous_time_model
    else:
        simplify_model = _simplify_discrete_time_model

    decomposition = []
    for z, p, k in rf.summation_decomposition(*args, tol=tol, **kwargs):
        z, p, k = simplify_model(z, p, k, tol)
        z = poly.real_polynomial_roots_if_close(z, tol)
        decomposition.append((z, p, k))
    return decomposition

@model_decomposition
def series_decomposition(*args, dt, **kwargs):
    # TODO(hidmic): support MIMO systems
    return rf.real_product_decomposition(*args, **kwargs)

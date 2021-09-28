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


def simple_if_possible(value, tol=1e-16, copy=False):
    if copy or np.isscalar(value):
        value = np.array(value)
    close_to_zero = np.abs(value) < tol
    value[close_to_zero] = 0.0
    if not np.all(np.isreal(value)):
        close_to_real = np.abs(np.imag(value)) < tol
        value[close_to_real] = np.real(value[close_to_real])
        close_to_imag = np.abs(np.real(value)) < tol
        value[close_to_imag] = 1j * np.imag(value[close_to_imag])
    return value


def asscalar_if_possible(value):
    if not np.isscalar(value):
        if value.ndim == 0:
            return value.item()
    return value


def asvector_if_possible(value):
    if np.ndim(value) == 2:
        if value.shape[0] == 1:
            return value[0]
        if value.shape[1] == 1:
            return value.T[0]
    return value


def vectorize(f=None, **kwargs):
    def __decorator(g):
        h = np.vectorize(g, **kwargs)

        @functools.wraps(g)
        def __wrapper(*args, **kwargs):
            ret = h(*args, **kwargs)
            if not isinstance(ret, tuple):
                return asscalar_if_possible(ret)
            return tuple(asscalar_if_possible(e) for e in ret)

        return __wrapper

    if f is not None:
        return __decorator(f)
    return __decorator


def empty_of(prototype, shape):
    if not isinstance(shape, tuple):
        shape = (shape,)
    return np.empty(shape=shape + prototype.shape, dtype=prototype.dtype)


def split(array, indices):
    if len(indices) > 0:
        yield array[: indices[0]]
        if len(indices) > 1:
            for start, end in zip(indices[0:-1], indices[1:]):
                yield array[start:end]
        yield array[indices[-1] :]
    else:
        yield array


@vectorize(excluded="set_")
def within(value, set_):
    return value in set_

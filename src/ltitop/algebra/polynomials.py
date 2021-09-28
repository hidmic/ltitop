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

from functools import reduce
from itertools import chain, repeat

import numpy as np

from ltitop.common.arrays import simple_if_possible


def from_roots(r, m=None):
    if m is not None:
        r = list(chain(*(repeat(ri, mi) for ri, mi in zip(r, m))))
    return np.poly(r)


def simplify(poly, tol=1e-16):
    poly = np.asarray(poly)
    close_to_real = np.abs(np.imag(poly)) < tol
    poly[close_to_real] = np.real(poly[close_to_real])
    close_to_zero = np.abs(poly) < tol
    poly[close_to_zero] = 0
    if np.count_nonzero(poly) == 0:
        return poly[0:1]
    return np.trim_zeros(poly, trim="f")


add = np.polyadd


def summation(polys):
    return reduce(add, polys)


multiply = np.polymul

divide = np.polydiv


def product(polys):
    return reduce(multiply, polys)


evaluate = np.polyval


def real_polynomial_roots_if_close(roots, tol=1e-16, rtype="avg"):
    if rtype in ["max", "maximum"]:
        align = np.max
    elif rtype in ["min", "minimum"]:
        align = np.min
    elif rtype in ["avg", "mean"]:
        align = np.mean
    else:
        raise ValueError(
            "`rtype` must be one of "
            "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}"
        )

    r = np.asarray(roots)
    indices = list(range(len(roots)))
    while indices:
        i = indices.pop()
        r[i] = simple_if_possible(r[i], tol)
        if not np.isreal(r[i]):
            if not indices:
                raise ValueError(f"No complex conjugate pole for {r[i]}")
            ri_star = np.conj(r[i])
            deviations = np.abs(r[indices] - ri_star)
            smallest_deviations = np.ma.masked_greater(deviations, tol, copy=False)
            if smallest_deviations.count() == 0:
                closest_root = r[indices[np.argmin(deviations)]]
                raise ValueError(
                    f"No complex conjugate root for {r[i]},"
                    f" closest root to conjugate is {closest_root}"
                )
            j = indices.pop(np.argmin(smallest_deviations))
            r[j] = align([ri_star, r[j]])
            r[i] = np.conj(r[j])
    return r


def real_polynomial_factorization_roots(roots, tol=1e-16):
    factors = []
    roots = list(roots)
    while roots:
        r = roots.pop()
        if np.abs(np.imag(r)) < tol:
            r = np.real(r)
        if not np.isreal(r):
            r_star = np.conj(r)
            if r_star not in roots:
                roots = np.asarray(roots)
                closest = roots[np.argmin(np.abs(roots - r_star))]
                raise ValueError(
                    f"No complex conjugate root for {r},"
                    f" closest root to conjugate is {closest}"
                )
            factors.append((r, r_star))
            roots.remove(r_star)
        else:
            factors.append((r,))
    return factors


def real_polynomial_factorization(poly, tol=1e-16):
    factor_roots = real_polynomial_factorization_roots(np.roots(poly), tol=tol)
    return poly[0], list(map(from_roots, factor_roots))

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
from functools import reduce

import itertools
import random
import math

import numpy as np
import scipy.linalg
import scipy.misc
import scipy.signal as signal
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import autograd as auto

import ltitop.algebra.polynomials as poly
from ltitop.common.arrays import simple_if_possible
from ltitop.common.arrays import asscalar_if_possible
from ltitop.common.arrays import split
from ltitop.common.arrays import vectorize


def add(z0, p0, k0, z1, p1, k1):
    num = poly.simplify(poly.add(
        k0 * poly.from_roots(np.concatenate((z0, p1))),
        k1 * poly.from_roots(np.concatenate((z1, p0)))
    ))
    return np.roots(num), np.concatenate((p0, p1)), num[0]


def summation(zpks):
    return reduce(lambda zpk0, zpk1: add(*zpk0, *zpk1), zpks)


def multiply(z0, p0, k0, z1, p1, k1):
    z = np.concatenate((z0, z1))
    p = np.concatenate((p0, p1))
    k = k0 * k1
    return z, p, k


def product(zpks):
    return reduce(lambda zpk0, zpk1: multiply(*zpk0, *zpk1), zpks)


@vectorize(excluded=[0, 1, 2])
def evaluate(z, p, k, x):
    return k * np.prod(x - z) / np.prod(x - p)


def partial_fractions_expansion(z, p, tol=1e-16, rtype='avg'):
    if len(z) == 0 and len(p) == 0:
        return z, p, np.array([])

    if len(z) >= len(p):
        raise ValueError('Cannot expand improper rational function')

    if len(p) == 1:
        return np.array([1]), p, np.array([1])

    # Use modified Brugia method
    u, su = signal.unique_roots(z, tol=tol, rtype=rtype)
    v, sv = signal.unique_roots(p, tol=tol, rtype=rtype)
    N = lambda x: np.prod((x - u)**su)

    n = 0
    r = np.zeros(len(p), dtype=complex)
    vm = np.ma.array(v, mask=np.zeros_like(v))
    for j, (vj, sj) in enumerate(zip(v, sv)):
        vm.mask[j] = True
        Dj = lambda x: np.prod((x - vm)**sv)
        r[n] = N(vj) / Dj(vj)
        if sj > 1:
            h = lambda r: np.sum(su / (vj - u)**r) - np.sum(sv / (vj - vm)**r)
            b = np.c_[[h(i) for i in range(1, sj)]]
            A = np.zeros((sj - 1, sj - 1), dtype=complex)
            for i in range(1, sj):
                A[i - 1, i - 1] = i
                A[i:, i - 1] = b[:-i]
                A[i - 1:, i - 1] *= -1**(i - 1)
            c = scipy.linalg.solve_triangular(A, b, lower=True)
            r[n + 1:n + sj] = c * r[n]
        vm.mask[j] = False
        n += sj

    r = simple_if_possible(r, tol)
    p = np.concatenate([[vi] * si for vi, si in zip(v, sv)])
    s = np.concatenate([range(1, si + 1) for si in sv])
    return r, p, s


def partial_fractions_companion_pencil(r, p, s):
    d = len(p) - 1
    C0 = np.zeros((d + 2, d + 2), dtype=complex)
    C0[-1, :-1] = -r / np.max(np.abs(r))
    C0[np.argwhere(s == 1), -1] = 1.
    C0[np.diag_indices(d + 1)] = p
    i = np.argwhere(s != 1); j = i - 1
    C0[i, j] = 1.
    C1 = np.eye(d + 2)
    C1[-1, -1] = 0
    return C0, C1


def partial_fractions_roots(r, p, s, tol=1e-16):
    if not np.any(r):
        return []
    C0, C1 = partial_fractions_companion_pencil(r, p, s)
    z = scipy.linalg.eig(C0, b=C1, right=False)
    # Drop spurious zeros near infinity
    z = z[np.isfinite(z)]
    z = simple_if_possible(z, tol)
    return z


def summation_decomposition(z, p, k, nterms=2, variant=0, tol=1e-10, rtype='avg'):
    direct_terms = None
    if len(z) >= len(p):
        # NOTE(hidmic): monomial bases are numerically
        # ill-conditioned, this will destroy all functions
        # but the lowest order ones
        # TODO(hidmic): can polynomial division be defined
        # for non-monomial bases?
        direct_terms, r = poly.divide(
            poly.from_roots(z),
            poly.from_roots(p))
        z = np.roots(r)

    r, p, s = partial_fractions_expansion(z, p, tol, rtype)

    if rtype in ['max', 'maximum']:
        align = np.max
    elif rtype in ['min', 'minimum']:
        align = np.min
    elif rtype in ['avg', 'mean']:
        align = np.mean
    else:
        raise ValueError(
            "`rtype` must be one of "
            "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}"
        )

    rd = random.Random(variant)
    indices = list(range(len(p)))
    indices_per_term = [[] for _ in range(nterms)]
    while indices:
        i = rd.randint(0, nterms - 1)
        j = indices.pop()
        p[j] = simple_if_possible(p[j], tol)
        if not np.isreal(p[j]):
            if not indices:
                raise ValueError(
                    f'No complex conjugate pole for {p[j]}')
            pj_star = np.conj(p[j])
            pp = p[indices]; ss = s[indices]
            deviations = np.abs(pp[ss == s[j]] - pj_star)
            smallest_deviations = \
                np.ma.masked_greater(deviations, tol, copy=False)
            if smallest_deviations.count() == 0:
                closest_pole = pp[np.argmin(deviations)]
                raise ValueError(
                    f'No complex conjugate pole for {p[j]},'
                    f' closest pole to conjugate is {closest_pole}')
            m = indices.pop(np.argmin(smallest_deviations))
            p[m] = align([pj_star, p[m]])
            p[j] = np.conj(p[m])
            indices_per_term[i].append(m)
        indices_per_term[i].append(j)

    terms = []
    for indices in indices_per_term:
        nth_term_residues = r[indices]
        nth_term_poles = p[indices]
        nth_term_order = s[indices]
        nth_term_zeros = partial_fractions_roots(
            nth_term_residues, nth_term_poles, nth_term_order, tol
        )
        nth_term_gain = k if np.any(nth_term_residues) else 0.
        terms.append((nth_term_zeros, nth_term_poles, nth_term_gain))
    if direct_terms:  # append them to the first term
        z, p, k = terms[0]
        z = np.roots(poly.add(
            poly.multiply(
                poly.from_roots(p),
                direct_terms
            ), poly.from_roots(z)))
        terms[0] = z, p, k
    return terms


def real_product_decomposition(zeros, poles, gain, nterms=2, variant=0, tol=1e-16):
    r = random.Random(variant)

    pole_groups = poly.real_polynomial_factorization_roots(poles, tol=tol)
    if len(pole_groups) < nterms:
        pole_groups.extend([()] * (nterms - len(pole_groups)))
    r.shuffle(pole_groups)
    pole_groups_per_term = split(pole_groups, sorted(
        r.sample(range(1, len(pole_groups)), nterms - 1)
    ))
    poles_per_term = list(map(np.concatenate, pole_groups_per_term))

    zero_groups = poly.real_polynomial_factorization_roots(zeros, tol=tol)
    if len(zeros) <= len(poles):  # preserve property
        zeros_per_term = [[] for _ in range(nterms)]
        if len(zero_groups) > 0:
            zero_groups = sorted(zero_groups, key=len, reverse=True)

            ranks = [[] for _ in range(len(zero_groups[0]))]
            for i, poles in enumerate(poles_per_term):
                for k in range(min(len(poles), len(ranks))):
                    ranks[k].append(i)

            for zeros in zero_groups:
                ki = len(zeros)
                i = r.choice(ranks[ki - 1])
                zeros_per_term[i].extend(zeros)
                ko = len(poles_per_term[i]) - len(zeros_per_term[i])
                for k in range(min(ko, len(ranks)), ki):
                    ranks[k].remove(i)
    else:
        if len(zero_groups) < nterms:
            zero_groups.extend([()] * (nterms - len(zero_groups)))
        r.shuffle(zero_groups)
        zero_groups_per_term = split(zero_groups, sorted(
            r.sample(range(1, len(zero_groups)), nterms - 1)
        ))
        zeros_per_term = list(map(np.concatenate, zero_groups_per_term))

    gain_per_term = np.full(nterms, np.power(np.absolute(gain), 1.0 / nterms))
    if gain < 0:
        negative_nterms = 2 * r.randint(0, math.floor((nterms - 1) / 2)) + 1
        gain_per_term[:negative_nterms] = -gain_per_term[:negative_nterms]
        r.shuffle(gain_per_term)

    return tuple(zip(zeros_per_term, poles_per_term, gain_per_term))

# NOTE(hidmic): EXPERIMENTAL

def complex_taylor_series_expansion(f, z, s, dx=1e-6):
    f = [f]
    s_max = np.max(s)
    for i in range(1, s_max):
        # TODO(hidmic): figure out why automatic
        # differentiation hangs (recursion?)
        # f.append(auto.holomorphic_grad(f[-1]))
        f.append(functools.partial(
            scipy.misc.derivative, f[0],
            dx=dx, n=i, order=2 * i + 1))
    F = np.full(
        (len(z), s_max),
        fill_value=np.nan,
        dtype=complex
    )
    for i, (zi, si) in enumerate(zip(z, s)):
        for k in range(si):
            F[i, k] = f[k](zi) / math.factorial(k)
    return F


def jordan_block(z, s):
    J = z * np.eye(s)
    if s > 1:
        J[:-1, 1:] += np.eye(s - 1)
    return J


def hermite_companion_pencil(f, alpha, z, s):
    if callable(f):
        f = complex_taylor_series_expansion(f, z, s)
    d = np.sum(s) - 1
    C0 = np.zeros((d + 2, d + 2), dtype=complex)
    C0[0, 1:] = np.concatenate([
        alpha[i, :si] for i, si in enumerate(s)
    ])
    C0[0, 1:] = C0[0, 1:] / np.max(np.abs(C0[0, 1:]))
    C0[1:, 0] = np.concatenate([
        f[i, :si] for i, si in enumerate(s)
    ])
    C0[1:, 0] = C0[1:, 0] / np.max(np.abs(C0[1:, 0]))
    C0[1:, 1:] = scipy.linalg.block_diag(*(
        jordan_block(zi, si).T for zi, si in zip(z, s)
    ))
    C1 = np.eye(d + 2)
    C1[0, 0] = 0
    return C0, C1


def newtonian_interpolant_denominator(f, z, s, m, k):
    v = np.zeros(np.sum(s), dtype=complex)
    if k > 0:
        if callable(f):
            f = complex_taylor_series_expansion(f, z, s)
        t = np.concatenate([[zi] * si for zi, si in zip(z, s)])
        ind = np.concatenate([[i] * si for i, si in enumerate(s)])
        ddiff_table = np.zeros((m + 2, m + 2), dtype=complex)
        ddiff_table[:m + 1, 0] = f[ind[:m + 1], 0]
        for j in range(1, m + 1):
            for i in range(m + 1 - j):
                if t[i] != t[i + j]:
                    ddiff_table[i, j] = (
                        ddiff_table[i + 1, j - 1] -
                        ddiff_table[i, j - 1]
                    ) / (t[i + j] - t[i])
                else:
                    ddiff_table[i, j] = f[ind[i], j]

        Amk = np.zeros((k, k + 1), dtype=complex)
        for i in range(1, k + 1):
            ddiff_table[m + 1, 0] = f[ind[m + i], 0]
            for j in range(1, m + 2):
                if t[m + 1 - j] != t[m + i]:
                    ddiff_table[m + 1 - j, j] = (
                        ddiff_table[m + 2 - j, j - 1] -
                        ddiff_table[m + 1 - j, j - 1]
                    ) / (t[m + i] - t[m + 1 - j])
                else:
                    ddiff_table[m + 1 - j, j] = f[ind[m + i], j]
            Amk[i - 1, :] = np.diagonal(np.fliplr(ddiff_table))[:k + 1]

        _, _, U = scipy.linalg.lu(Amk)
        indices, = np.where(np.diagonal(U) == 0.)
        d = indices[0] if len(indices) > 0 else k
        v[:d] = np.linalg.solve(U[:d, :d], -U[:d, d])
    else:
        d = k
    v[d] = 1.
    base = 0
    beta = np.full(
        (len(z), np.max(s)),
        fill_value=np.nan,
        dtype=complex
    )
    for i, si in enumerate(s):
        beta[i, :si] = v[base:base + si]
        base = base + si
    return d, beta


def newtonian_to_lagrange_hermite(beta, z, s):
    """
    For more information, see
      | Hermite interpolation: the barycentric approach
      | C. Schneider and W. Werner. 1991.
      | Computing 46, 1 (1991), 35–51.
      | DOI: https://doi.org/10.1007/BF02239010
    """
    n = len(z) - 1
    alpha = np.full(
        (n + 1, np.max(s)),
        np.nan, dtype=complex
    )
    beta = np.array(beta)
    for k in range(n):
        for m in range(k + 1, n + 1):
            a = 1 / (z[k] - z[m])
            for i in range(s[m]):
                beta[k, 0] = a * beta[k, 0]
                for j in range(1, s[k]):
                    beta[k, j] = a * (beta[k, j] - beta[k, j - 1])
                beta[m, i] = beta[m, i] - beta[k, s[k] - 1]
        alpha[k, :s[k]] = np.flip(beta[k, :s[k]])
    alpha[n, :s[n]] = np.flip(beta[n, :s[n]])
    return alpha


def hermite_rational_interpolant(f, z, s, m, k):
    if m < k:
        raise ValueError(f'Prescribed numerator degree ({m}) is'
                         f' lower than denominator degree ({k})')
    if np.sum(s) - 1 < m + k:
        raise ValueError(f'Not enough support points')
    if callable(f):
        f = complex_taylor_series_expansion(f, z, s)
    d, beta = newtonian_interpolant_denominator(f, z, s, m, k)
    alpha = newtonian_to_lagrange_hermite(beta, z, s)
    C0, C1 = hermite_companion_pencil(f, alpha, z, s)
    zeros, _ = scipy.linalg.eig(C0, b=C1)
    zeros = zeros[np.argsort(np.abs(zeros))[:m]]
    zeros = np.real_if_close(zeros)
    C0[1:, 0] = 1
    poles, _ = scipy.linalg.eig(C0, b=C1)
    poles = poles[np.argsort(np.abs(poles))[:d]]
    poles = np.real_if_close(poles)
    n = len(z) - 1
    kd = np.eye(n + 1, dtype=int)
    gain = poly.summation(
        alpha[i, p] * f[i, j] * poly.from_roots(
            z, s + kd[i] * (j - p - 1)
        )
        for i in range(n + 1)
        for p in range(s[i])
        for j in range(p + 1)
    )[n - m]
    gain = np.real_if_close(gain)
    return zeros, poles, gain


def real_summation_partition(zpk, variant=0, tol=1e-16):
    r = random.Random(variant)
    zeros, poles, gain = zpk

    pole_groups = \
        poly.real_polynomial_factorization_roots(poles, tol=tol)
    if len(pole_groups) < 2:
        return rf, ((), (), 0.)  # degenerate case
    r.shuffle(pole_groups)
    i = r.choice(range(1, len(pole_groups)))
    q = np.concatenate(pole_groups[:i])
    u = np.concatenate(pole_groups[i:])
    if len(q) < len(u):
        q, u = u, q  # improper rational for interpolation
    def f(z):
        return -(reduce(auto.numpy.multiply, z - q, 1.) /
                 reduce(auto.numpy.multiply, z - u, 1.))
    z, s = signal.unique_roots(zeros, tol=tol, rtype='avg')
    n = np.sum(s) - 1
    k = math.floor(n / (1 + len(q) / len(u))); m = n - k
    p, t, v = hermite_rational_interpolant(f, z, s, m, k)
    p = poly.real_polynomial_roots_if_close(p, tol=tol)
    t = poly.real_polynomial_roots_if_close(t, tol=tol)
    kp, kt = np.abs(v), np.sign(v)
    kc = gain / poly.simplify(poly.add(
        kp * poly.from_roots(np.concatenate((p, u))),
        kt * poly.from_roots(np.concatenate((t, q)))
    ))[0]
    rfL = p, q, kc * kp
    rfR = t, u, kc * kt
    return rfL, rfR

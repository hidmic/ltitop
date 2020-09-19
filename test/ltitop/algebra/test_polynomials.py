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

import ltitop.algebra.polynomials as poly

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
import pytest


def test_from_roots():
    # P = z + 1
    assert_almost_equal([1, 1], poly.from_roots((-1,)))
    # P = z - 1
    assert_almost_equal([1, -1], poly.from_roots((1,)))
    # P = (z - 1)^2 = (z^2 - 2 * z + 1)
    assert_almost_equal([1, -2, 1], poly.from_roots((1, 1)))
    # P = (z - 2) * (z - 5) = (z^2 - 7 * z + 10)
    assert_almost_equal([1, -7, 10], poly.from_roots((2, 5)))
    # P = (z - (0.5+0.5j)) * (z - (0.5-0.5j)) = (z^2 - z + 0.5)
    assert_almost_equal([1, -1, 0.5], poly.from_roots((0.5+0.5j, 0.5-0.5j)))

def test_simplify():
    # P = z^3 + z^2 + 1e-6 * z + (1+1e-6j) ~ z^3 + z^2 + 1
    assert_equal([1, 1, 0, 1], poly.simplify([1, 1, 1e-6, 1+1e-6j], tol=1e-5))

def test_add():
    # P = z^2 - z + 1
    P = [1, -1, 1]
    # Q = z
    Q = [1, 0]
    # R = P + Q = z^2 + 1
    R = [1, 0, 1]
    assert_equal(R, poly.add(P, Q))

def test_summation():
    # P = z^2 - z + 1
    P = [1, -1, 1]
    # Q = z
    Q = [1, 0]
    # R = 2 * z^3 - 2 * z + 5
    R = [2, 0, -2, 5]
    # S = P + Q + R = 2 * z^3 + z^2 - 2 * z + 6
    S = [2, 1, -2, 6]
    assert_equal(S, poly.summation([P, Q, R]))

def test_multiply():
    # P = z - 2
    P = [1, -2]
    # Q = z^2 - 1
    Q = [1, 0, -1]
    # R = P * Q = z^3 - 2 * z^2 - z + 2
    R = [1, -2, -1, 2]
    assert_equal(R, poly.multiply(P, Q))

def test_product():
    # P = z - 2
    P = [1, -2]
    # Q = z^2 - 1
    Q = [1, 0, -1]
    # R = z^3 + z
    R = [1, 0, 1, 0]
    # S = P * Q * R = z^6 - 2 * z^5 - z^4 + 2 * z^3    z^4 - 2 * z^3 - z^2 + 2 * z
    S = [1, -2, 0, 0, -1, 2, 0]
    assert_equal(S, poly.product([P, Q, R]))

def test_real_polynomial_factorization():
    # P = z
    P = [1, 0]
    kf, fterms = poly.real_polynomial_factorization(P)
    assert_almost_equal(kf, 1)
    assert len(fterms) == 1
    assert_almost_equal(fterms[0], P)

    # P = k * (z - 1) * (z - 1) * (z + 5) * (z^2 + z + 1)
    kp = 10
    pterms = sorted([[1, -1], [1, -1], [1, 5], [1, 1, 1]])
    P = kp * poly.product(pterms)

    kf, fterms = poly.real_polynomial_factorization(P, tol=1e-6)
    assert_almost_equal(kf, kp)
    assert len(pterms) == len(fterms)
    fterms = sorted(map(list, fterms))
    for pterm, fterm in zip(pterms, fterms):
        assert_almost_equal(pterm, fterm)


def test_complex_polynomial_factorization():
    # P = k * (z - 2) * (z - 1+1j)
    kp = 1
    pterms = [[1, -1], [1, -1+1j]]
    P = kp * poly.product(pterms)

    with pytest.raises(ValueError):
        poly.real_polynomial_factorization(P)

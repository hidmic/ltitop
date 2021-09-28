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

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.signal import tf2zpk, zpk2tf

import ltitop.algebra.rational_functions as rf


def test_add():
    # F = (z + 1) / (z^3 - 3 * z^2 + 10 * z + 1)
    F = tf2zpk([1, 1], [1, -3, 10, 1])
    # G = 1 / (z^2 - z + 3)
    G = tf2zpk([1], [1, -1, 3])
    # H = F + G = \
    #        (2 * z^3 - 3 * z^2 + 12 * z + 4) /
    # (z^5 - 4 * z^4 + 16 * z^3 - 18 * z^2 + 29 * z + 3)
    Hz, Hp, Hk = tf2zpk([2, -3, 12, 4], [1, -4, 16, -18, 29, 3])
    Sz, Sp, Sk = rf.add(*F, *G)
    assert_almost_equal(np.sort(Hz), np.sort(Sz))
    assert_almost_equal(np.sort(Hp), np.sort(Sp))
    assert_almost_equal(Hk, Sk)


def test_summation():
    # F = (z + 1) / (z^3 - 3 * z^2 + 10 * z + 1)
    F = tf2zpk([1, 1], [1, -3, 10, 1])
    # G = 1 / (z^2 - z + 3)
    G = tf2zpk([1], [1, -1, 3])
    # H = (-z^2 + 1) / 1
    H = tf2zpk([-1, 0, 1], [1])
    # J = F + G + H = \
    # (-z^7 + 4 * z^6 - 15 * z^5 + 14 * z^4 - 13 * z^3 - 21 * z^2 + 29 * z + 3) /
    #         (z^5 - 4 * z^4 + 16 * z^3 - 18 * z^2 + 29 * z + 3)
    Jz, Jp, Jk = tf2zpk([-1, 4, -15, 14, -11, -24, 41, 7], [1, -4, 16, -18, 29, 3])
    Sz, Sp, Sk = rf.summation([F, G, H])
    assert_almost_equal(np.sort(Jz), np.sort(Sz))
    assert_almost_equal(np.sort(Jp), np.sort(Sp))
    assert_almost_equal(Jk, Sk)


def test_multiply():
    # F = (z + 1) / (z^3 - 3 * z^2 + 10 * z + 1)
    F = tf2zpk([1, 1], [1, -3, 10, 1])
    # G = 1 / (z^2 - z + 3)
    G = tf2zpk([1], [1, -1, 3])
    # H = F * G = \
    #                     (z + 1) /
    # (z^5 - 4 * z^4 + 16 * z^3 - 18 * z^2 + 29 * z + 3)
    Hz, Hp, Hk = tf2zpk([1, 1], [1, -4, 16, -18, 29, 3])
    Mz, Mp, Mk = rf.multiply(*F, *G)
    assert_almost_equal(np.sort(Hz), np.sort(Mz))
    assert_almost_equal(np.sort(Hp), np.sort(Mp))
    assert_almost_equal(Hk, Mk)


def test_product():
    # F = (z + 1) / (z^3 - 3 * z^2 + 10 * z + 1)
    F = tf2zpk([1, 1], [1, -3, 10, 1])
    # G = 1 / (z^2 - z + 3)
    G = tf2zpk([1], [1, -1, 3])
    # H = (-5 * z^2 + 10) / 2
    H = tf2zpk([-5, 0, 10], [2])
    # J = F * G * H = \
    #         (-5 * z^3 - 5 * z^2 + 10 z + 10) /
    # (2 * z^5 - 8 * z^4 + 32 * z^3 - 36 * z^2 + 58 * z + 6)
    Jz, Jp, Jk = tf2zpk([-5, -5, 10, 10], [2, -8, 32, -36, 58, 6])
    Mz, Mp, Mk = rf.product([F, G, H])
    assert_almost_equal(np.sort(Jz), np.sort(Mz))
    assert_almost_equal(np.sort(Jp), np.sort(Mp))
    assert_almost_equal(Jk, Mk)


def test_partial_fractions_expansion():
    # F = z / (z^3 + 2 * z^2 + 5 * z + 4)
    #   = (0.125 - 0.22592403j) / (z + 0.5 - 1.93649167j) +
    #     (0.125 + 0.22592403j) / (z + 0.5 + 1.93649167j) +
    #     (-0.25) / (z + 1)
    Fz, Fp, Fk = tf2zpk([1, 0], [1, 2, 5, 4])
    residues, poles, order = rf.partial_fractions_expansion(Fz, Fp, tol=1e-10)

    expected_residues = np.array([-0.25, 0.125 - 0.22592403j, 0.125 + 0.22592403j])
    expected_poles = np.array([-1.0, -0.5 + 1.93649167j, -0.5 - 1.93649167j])
    expected_order = np.array([1, 1, 1])

    ordering = np.argsort(residues)
    assert_almost_equal(residues[ordering], expected_residues)
    assert_almost_equal(poles[ordering], expected_poles)
    assert_almost_equal(order[ordering], expected_order)


def test_summation_decomposition():
    # F = 1 / (z + 0.5)
    F = Fz, Fp, Fk = tf2zpk([1.0], [1.0, 0.5])
    terms = rf.summation_decomposition(*F, nterms=2)
    Sz, Sp, Sk = rf.summation(terms)
    assert_almost_equal(np.sort(Sz), np.sort(Fz))
    assert_almost_equal(np.sort(Sp), np.sort(Fp))
    assert_almost_equal(Sk, Fk)

    # F = (z^2 - 2) / (z^4 + 2 * z^3 + 2 * z^2 + 1)
    F = Fz, Fp, Fk = tf2zpk([1, 0, -2], [1, 2, 2, 1])
    terms = rf.summation_decomposition(*F, nterms=2)
    Sz, Sp, Sk = rf.summation(terms)
    assert_almost_equal(np.sort(Sz), np.sort(Fz))
    assert_almost_equal(np.sort(Sp), np.sort(Fp))
    assert_almost_equal(Sk, Fk)


def isreal(rf):
    b, a = zpk2tf(*rf)
    return np.all(np.isreal(np.real_if_close(b))) and np.all(
        np.isreal(np.real_if_close(a))
    )


@pytest.mark.skip
def test_real_summation_partition():
    # F = 1 / (z + 0.5)
    F = Fz, Fp, Fk = tf2zpk([1.0], [1.0, 0.5])
    L, R = rf.real_summation_partition(F)
    assert isreal(L) and isreal(R)
    Sz, Sp, Sk = rf.add(L, R)
    assert_almost_equal(np.sort(Fz), np.sort(Sz))
    assert_almost_equal(np.sort(Fp), np.sort(Sp))
    assert_almost_equal(Fk, Sk)

    # F = (z^2 - 2) / (z^3 + 2 * z^2 + 2 * z + 1)
    F = Fz, Fp, Fk = tf2zpk([1, 0, -2], [1, 2, 2, 1])
    L, R = rf.real_summation_partition(F)
    assert isreal(L) and isreal(R)
    Sz, Sp, Sk = rf.add(L, R)
    assert_almost_equal(np.sort(Fz), np.sort(Sz))
    assert_almost_equal(np.sort(Fp), np.sort(Sp))
    assert_almost_equal(Fk, Sk)

    # F = (z^4 + 1) / (z^4 + 2 * z^3 + 2 * z^2 + 1)
    F = Fz, Fp, Fk = tf2zpk([1, 0, 0, 0, 1], [1, 2, 2, 0, 1])
    L, R = rf.real_summation_partition(F)
    assert isreal(L) and isreal(R)
    Sz, Sp, Sk = rf.add(L, R)
    assert_almost_equal(
        np.sort(np.around(Fz, decimals=15)), np.sort(np.around(Sz, decimals=15))
    )
    assert_almost_equal(
        np.sort(np.around(Fp, decimals=15)), np.sort(np.around(Sp, decimals=15))
    )
    assert_almost_equal(Fk, Sk)


def test_real_product_decomposition():
    # F = 1 / (z + 0.5)
    F = Fz, Fp, Fk = tf2zpk([1.0], [1.0, 0.5])
    terms = rf.real_product_decomposition(*F, nterms=2)
    assert all(isreal(term) for term in terms)
    Pz, Pp, Pk = rf.product(terms)
    assert_almost_equal(np.sort(Fz), np.sort(Pz))
    assert_almost_equal(np.sort(Fp), np.sort(Pp))
    assert_almost_equal(Fk, Pk)

    # F = (z^2 - 2) / (z^4 + 2 * z^3 - 2 * z^2 + 2 * z + 1)
    F = Fz, Fp, Fk = tf2zpk([1, 0, -2], [1, 2, -2, 2, 1])
    terms = rf.real_product_decomposition(*F, nterms=2)
    assert all(isreal(term) for term in terms)
    Pz, Pp, Pk = rf.product(terms)
    assert_almost_equal(np.sort(Fz), np.sort(Pz))
    assert_almost_equal(np.sort(Fp), np.sort(Pp))
    assert_almost_equal(Fk, Pk)

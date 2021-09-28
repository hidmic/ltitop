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

import mpmath
import pytest

from ltitop.arithmetic.interval import interval


@pytest.fixture(params=[int, float, mpmath.mpf])
def scalar(request):
    return request.param


def test_interval_construction(scalar):
    with pytest.raises(TypeError):
        interval()

    with pytest.raises(TypeError):
        interval(upper_bound=scalar(0))

    iv = interval(scalar(0))
    assert iv.lower_bound == scalar(0)
    assert iv.lower_bound == iv.upper_bound
    assert type(iv.lower_bound) is scalar
    assert type(iv.upper_bound) is scalar

    iv = interval(scalar(-1), scalar(1))
    assert iv.lower_bound == scalar(-1)
    assert type(iv.lower_bound) is scalar
    assert iv.upper_bound == scalar(1)
    assert type(iv.upper_bound) is scalar


def test_interval_comparison(scalar):
    iv_a = interval(scalar(-10), scalar(10))
    assert iv_a == iv_a
    assert not iv_a != iv_a
    assert not (iv_a < iv_a)
    assert not (iv_a <= iv_a)
    assert not (iv_a > iv_a)
    assert not (iv_a >= iv_a)

    iv_b = interval(scalar(90), scalar(110))
    assert not iv_a == iv_b
    assert iv_a != iv_b
    assert iv_a < iv_b
    assert iv_a <= iv_b
    assert not iv_a > iv_b
    assert not iv_a >= iv_b

    iv_c = interval(scalar(-110), scalar(-90))
    assert not iv_a == iv_c
    assert iv_a != iv_c
    assert iv_a > iv_c
    assert iv_a >= iv_c
    assert not iv_a < iv_c
    assert not iv_a <= iv_c

    iv_d = interval(scalar(-20), scalar(20))
    assert not iv_a == iv_d
    assert iv_a != iv_d
    assert not (iv_a < iv_d)
    assert not (iv_a <= iv_d)
    assert not (iv_a > iv_d)
    assert not (iv_a >= iv_d)

    iv_e = interval(scalar(-1), scalar(1))
    assert not iv_a == iv_e
    assert iv_a != iv_e
    assert not (iv_a < iv_e)
    assert not (iv_a <= iv_e)
    assert not (iv_a > iv_e)
    assert not (iv_a >= iv_e)

    iv_f = interval(scalar(10), scalar(20))
    assert not iv_a == iv_f
    assert iv_a != iv_f
    assert not (iv_a < iv_f)
    assert iv_a <= iv_f
    assert not (iv_a > iv_f)
    assert not iv_a >= iv_f

    iv_g = interval(scalar(-20), scalar(-10))
    assert not iv_a == iv_g
    assert iv_a != iv_g
    assert not (iv_a > iv_g)
    assert iv_a >= iv_g
    assert not (iv_a < iv_g)
    assert not iv_a <= iv_g

    s_h = scalar(5)
    assert not iv_a == s_h
    assert iv_a != s_h
    assert not (iv_a < s_h)
    assert not (iv_a <= s_h)
    assert not (iv_a > s_h)
    assert not (iv_a >= s_h)

    s_i = scalar(10)
    assert not iv_a == s_i
    assert iv_a != s_i
    assert not (iv_a < s_i)
    assert iv_a <= s_i
    assert not (iv_a > s_i)
    assert not iv_a >= s_i

    s_j = scalar(-10)
    assert not iv_a == s_j
    assert iv_a != s_j
    assert not (iv_a > s_j)
    assert iv_a >= s_j
    assert not (iv_a < s_j)
    assert not iv_a <= s_j

    s_k = scalar(20)
    assert not iv_a == s_k
    assert iv_a != s_k
    assert iv_a < s_k
    assert iv_a <= s_k
    assert not iv_a > s_k
    assert not iv_a >= s_k

    s_l = scalar(-20)
    assert not iv_a == s_l
    assert iv_a != s_l
    assert iv_a > s_l
    assert iv_a >= s_l
    assert not iv_a < s_l
    assert not iv_a <= s_l

    iv_h = interval(s_h, s_h)
    assert iv_h == s_h
    assert not iv_h != s_h
    assert not (iv_h < s_h)
    assert iv_h <= s_h
    assert not (iv_h > s_h)
    assert iv_h >= s_h


def test_interval_arithmetic(scalar):
    iv_a = interval(scalar(-1), scalar(1))
    iv_b = interval(scalar(-2), scalar(2))
    assert -iv_a == interval(scalar(-1), scalar(1))
    assert -iv_b == interval(scalar(-2), scalar(2))
    assert iv_a + iv_b == interval(scalar(-3), scalar(3))
    assert iv_a - iv_b == interval(scalar(-3), scalar(3))
    assert iv_a * iv_b == interval(scalar(-2), scalar(2))
    assert iv_b / iv_a == interval(scalar(-2), scalar(2))
    if scalar is not mpmath.mpf:
        assert iv_b // iv_a == interval(scalar(-2), scalar(2))
    assert iv_b % iv_a == scalar(0)

    iv_c = interval(scalar(4), scalar(8))
    assert -iv_c == interval(scalar(-8), scalar(-4))
    assert iv_b + iv_c == interval(scalar(2), scalar(10))
    assert iv_b - iv_c == interval(scalar(-10), scalar(-2))
    assert iv_b * iv_c == interval(scalar(-16), scalar(16))
    assert iv_c / iv_b == interval(scalar(-4), scalar(4))
    if scalar is not mpmath.mpf:
        assert iv_c // iv_b == interval(scalar(-4), scalar(4))
    assert iv_c % iv_b == scalar(0)


def test_interval_bitwise():
    iv = interval(2, 3)
    assert iv << 1 == interval(4, 6)
    assert iv >> 1 == 1
    assert iv >> 2 == 0


def test_interval_membership(scalar):
    iv_a = interval(scalar(-1), scalar(1))
    assert iv_a in iv_a
    iv_b = interval(scalar(-2), scalar(2))
    assert iv_a in iv_b
    assert iv_b not in iv_a
    iv_c = interval(scalar(0), scalar(2))
    assert iv_c not in iv_a
    assert iv_c in iv_b
    iv_d = interval(scalar(-5), scalar(0))
    assert iv_d not in iv_a
    assert iv_d not in iv_b
    assert iv_d not in iv_c
    s_a = scalar(0)
    assert s_a in iv_a
    assert s_a in iv_b
    assert s_a in iv_c
    assert s_a in iv_d
    s_b = scalar(2)
    assert s_b not in iv_a
    assert s_b in iv_b
    assert s_b in iv_c
    assert s_b not in iv_d
    s_c = scalar(5)
    assert s_c not in iv_a
    assert s_c not in iv_b
    assert s_c not in iv_c
    assert s_c not in iv_d

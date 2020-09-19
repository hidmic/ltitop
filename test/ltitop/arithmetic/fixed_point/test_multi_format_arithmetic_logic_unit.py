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

from ltitop.arithmetic.errors import UnderflowError
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.modular import wraparound
from ltitop.arithmetic.rounding import nearest_integer
from ltitop.arithmetic.fixed_point.formats import uQ, Q

from ltitop.arithmetic.fixed_point.multi_format_arithmetic_logic_unit \
    import MultiFormatArithmeticLogicUnit

import pytest

def test_construction_errors():
    with pytest.raises(ValueError):
        MultiFormatArithmeticLogicUnit(wordlength=0)

def test_represent_errors():
    alu = MultiFormatArithmeticLogicUnit(
        wordlength=8,
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False
    )

    with pytest.raises(ValueError):
        alu.represent(-1, format_=uQ(8))

    with pytest.raises(UnderflowError):
        alu.represent(1e-3, format_=Q(7))

    with pytest.raises(OverflowError):
        alu.represent(10, format_=Q(7))

    with pytest.raises(ValueError):
        alu.represent(0, format_=Q(15))


def test_represent():
    alu = MultiFormatArithmeticLogicUnit(
        wordlength=8,
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False
    )

    assert alu.rinfo(signed=True).eps == 0.0078125
    assert alu.rinfo(signed=True).min == -128
    assert alu.rinfo(signed=True).max == 127
    assert alu.rinfo(signed=False).eps == 0.00390625
    assert alu.rinfo(signed=False).min == 0
    assert alu.rinfo(signed=False).max == 256

    r = alu.represent(0.5)
    assert r.mantissa == 64
    assert r.format_ == Q(7)

    r = alu.represent(0)
    assert r.mantissa == 0
    assert r.format_ == Q(7)

    r = alu.represent(1.25)
    assert r.mantissa == 80
    assert r.format_ == Q(2, 6)

    r = alu.represent(1.25, format_=Q(3, 4))
    assert r.mantissa == 20
    assert r.format_ == Q(3, 4)

    r = alu.represent(1.25, format_=uQ(3, 5))
    assert r.mantissa == 40
    assert r.format_ == uQ(3, 5)

    r = alu.represent(12.5)
    assert r.mantissa == 100
    assert r.format_ == Q(5, 3)

    r = alu.represent(12.5, format_=Q(6, 2))
    assert r.mantissa == 50
    assert r.format_ == Q(6, 2)

    r = alu.represent(12.5, format_=uQ(4, 4))
    assert r.mantissa == 200
    assert r.format_ == uQ(4, 4)

    r = alu.represent(15.6)
    assert r.mantissa == 125
    assert r.format_ == Q(5, 3)

    r = alu.represent(interval(-1, 1))
    assert r.mantissa == interval(-64, 64)
    assert r.format_ == Q(2, 6)

def test_add():
    alu = MultiFormatArithmeticLogicUnit(
        wordlength=8,
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False
    )

    x = alu.represent(1, format_=Q(4, 4))
    y = alu.represent(2, format_=Q(4, 4))
    z = alu.add(x, y)
    assert z.mantissa == 48
    assert z.format_ == Q(4, 4)

    x = alu.represent(1, format_=Q(2, 6))
    y = alu.represent(2, format_=Q(3, 5))
    z = alu.add(x, y)
    assert z.mantissa == 96
    assert z.format_ == Q(3, 5)

    x = alu.represent(1, format_=Q(2, 6))
    y = alu.represent(-2, format_=Q(3, 5))
    z = alu.add(x, y)
    assert z.mantissa == -32
    assert z.format_ == Q(3, 5)

    x = alu.represent(interval(-1, 1), format_=Q(2, 6))
    y = alu.represent(interval(3, 5), format_=Q(4, 4))
    z = alu.add(x, y)
    assert z.mantissa == interval(32, 96)
    assert z.format_ == Q(4, 4)

    x = alu.represent(interval(-1, 1), format_=Q(2, 6))
    y = alu.represent(interval(3, 5), format_=Q(4, 4))
    z = alu.add(x, y)
    assert z.mantissa == interval(32, 96)
    assert z.format_ == Q(4, 4)

def test_multiply():
    alu = MultiFormatArithmeticLogicUnit(
        wordlength=8,
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False
    )

    x = alu.represent(1, format_=Q(4, 4))
    y = alu.represent(2, format_=Q(4, 4))
    z = alu.multiply(x, y)
    assert z.mantissa == 64
    assert z.format_ == Q(3, 5)

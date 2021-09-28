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

import pytest

from ltitop.arithmetic.errors import OverflowError, UnderflowError
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit import (
    FixedFormatArithmeticLogicUnit,
)
from ltitop.arithmetic.fixed_point.formats import Q
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.modular import wraparound
from ltitop.arithmetic.rounding import nearest_integer


def test_represent_errors():
    alu = FixedFormatArithmeticLogicUnit(
        format_=Q(7),
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False,
    )

    with pytest.raises(UnderflowError):
        alu.represent(1e-3)

    with pytest.raises(OverflowError):
        alu.represent(10)


def test_represent():
    alu = FixedFormatArithmeticLogicUnit(
        format_=Q(7),
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False,
    )
    assert alu.rinfo().eps == 0.0078125
    assert alu.rinfo().min == -1
    assert alu.rinfo().max == 0.9921875

    r = alu.represent(0)
    assert r.mantissa == 0
    assert r.format_ == alu.format_

    r = alu.represent(0.25)
    assert r.mantissa == 32
    assert r.format_ == alu.format_

    r = alu.represent(-0.25)
    assert r.mantissa == -32
    assert r.format_ == alu.format_

    r = alu.represent(0.3)
    assert r.mantissa == 38
    assert r.format_ == alu.format_

    r = alu.represent(-0.3)
    assert r.mantissa == -38
    assert r.format_ == alu.format_

    r = alu.represent(interval(-0.75, 0.75))
    assert r.mantissa == interval(-96, 96)
    assert r.format_ == alu.format_


def test_add():
    alu = FixedFormatArithmeticLogicUnit(
        format_=Q(4, 4),
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=True,
        allows_underflow=False,
    )

    x = alu.represent(1)
    y = alu.represent(2)
    z = alu.add(x, y)
    assert z.mantissa == 48
    assert z.format_ == alu.format_

    x = alu.represent(4)
    z = alu.add(x, x)
    assert z.mantissa == -128
    assert z.format_ == alu.format_

    y = alu.represent(-2)
    z = alu.add(x, y)
    assert z.mantissa == 32
    assert z.format_ == alu.format_

    x = alu.represent(interval(-1, 1))
    y = alu.represent(interval(3, 5))
    z = alu.add(x, y)
    assert z.mantissa == interval(32, 96)
    assert z.format_ == alu.format_

    x = alu.represent(interval(-1, 3))
    y = alu.represent(interval(3, 5))
    z = alu.add(x, y)
    # NOTE(hidmic): this ALU handles overflow
    # in interval arithmetic by assuming the
    # value may be anywhere (how many cycles
    # does e.g. [32, -32] imply?)
    # TODO(hidmic): revisit result
    assert z.mantissa == interval(-128, 127)
    assert z.format_ == alu.format_


def test_multiply():
    alu = FixedFormatArithmeticLogicUnit(
        format_=Q(7),
        rounding_method=nearest_integer,
        overflow_behavior=wraparound,
        allows_overflow=False,
        allows_underflow=False,
    )

    x = alu.represent(0.5)
    y = alu.represent(0.5)
    z = alu.multiply(x, y)
    assert z.mantissa == 32
    assert z.format_ == alu.format_

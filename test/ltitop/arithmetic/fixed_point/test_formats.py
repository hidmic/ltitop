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

from ltitop.arithmetic.fixed_point.formats import Format
from ltitop.arithmetic.fixed_point.formats import uQ, sQ, Q
from ltitop.arithmetic.fixed_point.formats import uP, sP, P
import ltitop.arithmetic.rounding as rounding
from ltitop.arithmetic.interval import interval

import pytest

def test_signed_integer_formats():
    format_ = Format(msb=7, lsb=0, signed=True)
    assert format_.msb == 7
    assert format_.lsb == 0
    assert format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(-128, 127)
    assert format_.value_interval == interval(-128, 127)
    assert format_.value_epsilon == 1
    assert format_.to_qnotation() == 'Q8.0'
    assert format_.to_pnotation() == '(7,0)'

    mantissa, (underflow, overflow) = format_.represent(
        -127.5, rounding_method=rounding.nearest_integer
    )
    assert mantissa == -128
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(
        -127.5, rounding_method=rounding.ceil
    )
    assert mantissa == -127
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0.1)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(256)
    assert not underflow
    assert overflow

def test_unsigned_integer_formats():
    format_ = Format(msb=8, lsb=0, signed=False)
    assert format_.msb == 8
    assert format_.lsb == 0
    assert not format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(0, 255)
    assert format_.value_interval == interval(0, 255)
    assert format_.value_epsilon == 1
    assert format_.to_qnotation() == 'uQ8.0'
    assert format_.to_pnotation() == 'u(8,0)'

    with pytest.raises(ValueError):
        format_.represent(-1)

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0.1)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(256)
    assert not underflow
    assert overflow

def test_signed_scaled_integer_formats():
    format_ = Format(msb=7, lsb=2, signed=True)
    assert format_.msb == 7
    assert format_.lsb == 2
    assert format_.signed
    assert format_.wordlength == 6
    assert format_.mantissa_interval == interval(-32, 31)
    assert format_.value_interval == interval(-128, 124)
    assert format_.value_epsilon == 4
    assert format_.to_qnotation() == 'Q8.-2'
    assert format_.to_pnotation() == '(7,2)'

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(
        -17, rounding_method=rounding.nearest_integer
    )
    assert mantissa == -4  # i.e. -16
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(128)
    assert not underflow
    assert overflow

def test_unsigned_scaled_integer_formats():
    format_ = Format(msb=8, lsb=2, signed=False)
    assert format_.msb == 8
    assert format_.lsb == 2
    assert not format_.signed
    assert format_.wordlength == 6
    assert format_.mantissa_interval == interval(0, 63)
    assert format_.value_interval == interval(0, 252)
    assert format_.value_epsilon == 4
    assert format_.to_qnotation() == 'uQ8.-2'
    assert format_.to_pnotation() == 'u(8,2)'

    with pytest.raises(ValueError):
        format_.represent(-1)

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(128)
    assert mantissa == 32
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(256)
    assert not underflow
    assert overflow

def test_signed_fractional_formats():
    format_ = Format(msb=0, lsb=-7, signed=True)
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(-128, 127)
    assert format_.value_interval == interval(-1.0, 0.9921875)
    assert format_.value_epsilon == 0.0078125
    assert format_.to_qnotation() == 'Q1.7'
    assert format_.to_pnotation() == '(0,-7)'

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1e-6)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0.5)
    assert mantissa == 64
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1)
    assert not underflow
    assert overflow

def test_unsigned_fractional_formats():
    format_ = Format(msb=0, lsb=-8, signed=False)
    assert format_.msb == 0
    assert format_.lsb == -8
    assert not format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(0, 255)
    assert format_.value_interval == interval(0.0, 0.99609375)
    assert format_.value_epsilon == 0.00390625
    assert format_.to_qnotation() == 'uQ0.8'
    assert format_.to_pnotation() == 'u(0,-8)'

    with pytest.raises(ValueError):
        format_.represent(-0.1)

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1e-6)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0.5)
    assert mantissa == 128
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1)
    assert not underflow
    assert overflow

def test_signed_fixed_point_formats():
    format_ = Format(msb=3, lsb=-4, signed=True)
    assert format_.msb == 3
    assert format_.lsb == -4
    assert format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(-128, 127)
    assert format_.value_interval == interval(-8.0, 7.9375)
    assert format_.value_epsilon == 0.0625
    assert format_.to_qnotation() == 'Q4.4'
    assert format_.to_pnotation() == '(3,-4)'

    mantissa, (underflow, overflow) = format_.represent(-2.25)
    assert mantissa == -36
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1e-6)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(0.5)
    assert mantissa == 8
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(8)
    assert not underflow
    assert overflow

def test_unsigned_fixed_point_formats():
    format_ = Format(msb=4, lsb=-4, signed=False)
    assert format_.msb == 4
    assert format_.lsb == -4
    assert not format_.signed
    assert format_.wordlength == 8
    assert format_.mantissa_interval == interval(0, 255)
    assert format_.value_interval == interval(0.0, 15.9375)
    assert format_.value_epsilon == 0.0625
    assert format_.to_qnotation() == 'uQ4.4'
    assert format_.to_pnotation() == 'u(4,-4)'

    with pytest.raises(ValueError):
        format_.represent(-1)

    mantissa, (underflow, overflow) = format_.represent(0)
    assert mantissa == 0
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(1e-6)
    assert mantissa == 0
    assert underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(8)
    assert mantissa == 128
    assert not underflow
    assert not overflow

    mantissa, (underflow, overflow) = format_.represent(16)
    assert not underflow
    assert overflow

def test_invalid_formats():
    with pytest.raises(ValueError):
        Format(msb=0, lsb=2)

def test_q_notations():
    from_notation = Format.from_notation

    format_ = Q(15)
    assert format_ == Q(1, 15)
    assert format_.msb == 0
    assert format_.lsb == -15
    assert format_.signed
    assert format_ == from_notation('Q1.15')

    format_ = Q(1, 7)
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed
    assert format_ == from_notation('Q1.7')

    format_ = Q(3, -1)
    assert format_.msb == 2
    assert format_.lsb == 1
    assert format_.signed
    assert format_ == from_notation('Q3.-1')

    format_ = uQ(15)
    assert format_ == uQ(1, 15)
    assert format_.msb == 1
    assert format_.lsb == -15
    assert not format_.signed
    assert format_ == from_notation('uQ1.15')

    format_ = uQ(4, -2)
    assert format_.msb == 4
    assert format_.lsb == 2
    assert not format_.signed
    assert format_ == from_notation('uQ4.-2')

def test_parenthesis_notations():
    from_notation = Format.from_notation

    format_ = P(0, -15)
    assert format_.msb == 0
    assert format_.lsb == -15
    assert format_.signed
    assert format_ == from_notation('(0,-15)')

    format_ = P(0, -7)
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed
    assert format_ == from_notation('(0,-7)')

    format_ = P(2, 1)
    assert format_.msb == 2
    assert format_.lsb == 1
    assert format_.signed
    assert format_ == from_notation('(2,1)')

    format_ = uP(1, -15)
    assert format_.msb == 1
    assert format_.lsb == -15
    assert not format_.signed
    assert format_ == from_notation('u(1,-15)')

    format_ = uP(4, 2)
    assert format_.msb == 4
    assert format_.lsb == 2
    assert not format_.signed
    assert format_ == from_notation('u(4,2)')

def test_best_formats():
    mantissa, format_ = Format.best(
        0, wordlength=8, signed=True
    )
    assert mantissa == 0
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed

    mantissa, format_ = Format.best(
        0.5, wordlength=8, signed=True
    )
    assert mantissa == 64
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed

    mantissa, format_ = Format.best(
        1.0, wordlength=8, signed=True
    )
    assert mantissa == 64
    assert format_.msb == 1
    assert format_.lsb == -6
    assert format_.signed

    mantissa, format_ = Format.best(
        -1.0, wordlength=8, signed=True
    )
    assert mantissa == -128
    assert format_.msb == 0
    assert format_.lsb == -7
    assert format_.signed

    mantissa, format_ = Format.best(
        12.0, wordlength=8, signed=True
    )
    assert mantissa == 96
    assert format_.msb == 4
    assert format_.lsb == -3
    assert format_.signed

    mantissa, format_ = Format.best(
        1200.0, wordlength=8, signed=True
    )
    assert mantissa == 75
    assert format_.msb == 11
    assert format_.lsb == 4
    assert format_.signed

    mantissa, format_ = Format.best(
        interval(-16, 16), wordlength=8, signed=True
    )
    assert mantissa == interval(-64, 64)
    assert format_.msb == 5
    assert format_.lsb == -2
    assert format_.signed

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

from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.saturated import saturate


def test_saturate():
    value, overflow = saturate(32, range_=interval(0, 127))
    assert value == 32
    assert not overflow

    value, overflow = saturate(128, range_=interval(0, 127))
    assert value == 127
    assert overflow

    value, overflow = saturate(-1, range_=interval(0, 127))
    assert value == 0
    assert overflow

    value, overflow = saturate(-32, range_=interval(0, 127))
    assert value == 0
    assert overflow

    value, overflow = saturate(32, range_=interval(-128, 127))
    assert value == 32
    assert not overflow

    value, overflow = saturate(128, range_=interval(-128, 127))
    assert value == 127
    assert overflow

    value, overflow = saturate(-1, range_=interval(-128, 127))
    assert value == -1
    assert not overflow

    value, overflow = saturate(-32, range_=interval(-128, 127))
    assert value == -32
    assert not overflow

    value, overflow = saturate(-129, range_=interval(-128, 127))
    assert value == -128
    assert overflow

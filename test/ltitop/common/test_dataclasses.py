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

from dataclasses import FrozenInstanceError

import pytest

from ltitop.common.dataclasses import immutable_dataclass


def test_immutable_dataclass():
    @immutable_dataclass
    class Dummy:
        a: int
        b: str

    data = Dummy(1, "foo")
    assert data.a == 1
    assert data.b == "foo"

    with pytest.raises(FrozenInstanceError):
        data.a = 2

    with pytest.raises(FrozenInstanceError):
        data.b = "bar"

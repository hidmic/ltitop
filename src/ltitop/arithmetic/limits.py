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
import numpy as np

from ltitop.arithmetic.fixed_point.number import Number as FixedPointNumber
from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit
from ltitop.arithmetic.interval import Interval
from ltitop.common.dataclasses import immutable_dataclass


@immutable_dataclass
class mpinfo:
    eps: mpmath.mpf
    min: mpmath.mpf
    max: mpmath.mpf

    def __init__(self):
        super().__setattr__('eps', mpmath.eps)
        super().__setattr__('min', mpmath.ninf)
        super().__setattr__('max', mpmath.inf)


def info(dtype):
    witness = dtype(0)

    if isinstance(witness, mpmath.mpf):
        return mpinfo()

    if isinstance(witness, FixedPointNumber):
        return ProcessingUnit.active().rinfo()

    try:
        return np.iinfo(witness)
    except ValueError:
        pass

    try:
        return np.finfo(witness)
    except ValueError:
        pass

    raise TypeError(f'data type "{dtype}" not understood')


def interval(dtype):
    dinfo = info(dtype)
    return Interval(
        lower_bound=dinfo.min,
        upper_bound=dinfo.max
    )

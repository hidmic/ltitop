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
from ltitop.arithmetic.interval import Interval
from ltitop.arithmetic.saturated import saturate
from ltitop.common.arrays import vectorize


@vectorize(excluded=['range_'])
def wraparound(value, range_):
    if isinstance(value, Interval):
        # TODO(hidmic): use modulo interval arithmetic?
        lower_bound, lower_overflow = wraparound(
            value.lower_bound, range_=range_
        )
        upper_bound, upper_overflow = wraparound(
            value.upper_bound, range_=range_
        )
        overflow = np.logical_or(lower_overflow, upper_overflow)
        if not np.isscalar(overflow):
            if np.any(overflow):
                lower_bound[overflow] = range_.lower_bound
                upper_bound[overflow] = range_.upper_bound
        elif overflow:
            lower_bound = range_.lower_bound
            upper_bound = range_.upper_bound
        return Interval(lower_bound, upper_bound), overflow
    overflow = False
    if value < range_.lower_bound or value > range_.upper_bound:
        value = (value - range_.lower_bound) % (
            range_.upper_bound + 1 - range_.lower_bound
        ) + range_.lower_bound
        overflow = True
    return value, overflow

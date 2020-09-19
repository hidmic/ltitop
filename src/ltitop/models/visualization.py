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

import sympy
import sys

import scipy.signal.ltisys as ltisys


def pretty(model):
    if isinstance(model, ltisys.TransferFunction):
        if model.dt is None:  # continuous time
            s = sympy.Symbol('s')
            num = sum(c*s**i for i, c in enumerate(reversed(model.num)))
            den = sum(c*s**i for i, c in enumerate(reversed(model.den)))
        else:  # discrete time
            z = sympy.Symbol('z')
            num = sum(c*z**i for i, c in enumerate(reversed(model.num)))
            den = sum(c*z**i for i, c in enumerate(reversed(model.den)))
        return sympy.pretty(
            num / den, use_unicode=True,
            num_columns=sys.maxsize
        )
    raise TypeError(f'Unknown model type')

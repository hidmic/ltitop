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


class UnderflowError(ArithmeticError):

    def __init__(self, message='', value=None, epsilon=None):
        super().__init__(message)
        self.value = value
        self.epsilon = epsilon

    @property
    def margin(self):
        value = np.absolute(self.value)
        if isinstance(value, Interval):
            value = value.upper_bound - value.lower_bound
        return np.min(value / self.epsilon)


class OverflowError(ArithmeticError):

    def __init__(self, message='', value=None, limits=None):
        super().__init__(message)
        self.value = value
        self.limits = limits

    @property
    def margin(self):
        value = self.value
        if not isinstance(value, Interval):
            value = Interval(value)
        return np.min(np.array([
            self.limits.upper_bound - value.upper_bound,
            value.lower_bound - self.limits.lower_bound
        ]) / (self.limits.upper_bound - self.limits.lower_bound), axis=0)

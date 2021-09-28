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

from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit


class ArithmeticLogicUnit(ProcessingUnit):
    def __init__(self, *, wordlength, **kwargs):
        super().__init__(**kwargs)
        if wordlength < 1:
            raise ValueError(f"{wordlength} cannot be less than 1 bit")
        self.__wordlength = wordlength

    @property
    def wordlength(self):
        return self.__wordlength

    def __str__(self):
        return f"{self.wordlength} bits ALU"

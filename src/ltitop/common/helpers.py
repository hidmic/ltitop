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

import functools


def type_uniform_binary_operator(method):
    @functools.wraps(method)
    def __operator_wrapper(self, other):
        cls = type(self)
        if not isinstance(other, cls):
            other = cls(other)
        return cls(method(self, other))

    return __operator_wrapper


def astuple(value):
    try:
        return tuple(iter(value))
    except TypeError:
        return (value,)


def identity(x):
    return x


class methodcall:
    def __init__(self, instance, method_name):
        self.instance = instance
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        method = getattr(self.instance, self.method_name)
        return method(*args, **kwargs)

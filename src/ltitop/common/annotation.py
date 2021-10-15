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


def annotated_function(func=None, **annotations):
    class _wrapper:
        __slots__ = ("__func", "__annotations")

        def __init__(self, func):
            self.__func = func
            self.__annotations = annotations

        def __call__(self, *args, **kwargs):
            return self.__func(*args, **kwargs)

        def __getattr__(self, name):
            if name in self.__annotations:
                return self.__annotations[name]
            return getattr(self.__func, name)

        def annotate(self, *args, **kwargs):
            annotations = [(value.__name__, value) for value in args]
            annotations.extend(kwargs.items())
            for name, value in annotations:
                if name in self.__annotations:
                    raise ValueError(f"{name} already present")
                self.__annotations[name] = value

    if func is None:
        return _wrapper

    return _wrapper(func)

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

from abc import ABCMeta
import contextlib
import functools
import types


class Traceable(ABCMeta):

    @classmethod
    def wrap_init(cls, init):
        def __init__(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._traces = []
        return __init__

    @classmethod
    def wrap_method(cls, method):
        @functools.wraps(method)
        def decorator(self, *args, **kwargs):
            ret = method(self, *args, **kwargs)
            for trace in self._traces:
                trace.append((decorator, ret, args, kwargs))
            return ret
        return decorator

    @classmethod
    def make_public_api(cls):
        @contextlib.contextmanager
        def trace(self):
            trace = []
            self._traces.append(trace)
            try:
                yield trace
            finally:
                self._traces.pop()

        @contextlib.contextmanager
        def notrace(self):
            traces = self._traces
            self._traces = []
            try:
                yield
            finally:
                self._traces = traces

        return {'trace': trace, 'notrace': notrace}

    def __new__(cls, name, bases, dct):
        changes = {}
        for name in dct:
            if name.startswith('_'):
                continue
            if not isinstance(dct[name], types.FunctionType):
                continue
            changes[name] = cls.wrap_method(dct[name])
        if any(changes):
            if not any(isinstance(base, cls) for base in bases):
                init = base[0].__init__ if bases else object.__init__
                dct['__init__'] = cls.wrap_init(dct.get('__init__', init))
                dct.update(cls.make_public_api())
            dct.update(changes)
        return super().__new__(cls, name, bases, dct)

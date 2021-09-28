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

import dataclasses
import typing


class Dataclass(typing.Protocol):
    __dataclass_fields__: typing.Dict


def immutable_dataclass(cls=None, /, iterable=False, **kwargs):
    def _decorate(cls):
        cls = dataclasses.dataclass(cls, frozen=True, **kwargs)
        cls_dict = dict(cls.__dict__)
        field_names = tuple(f.name for f in dataclasses.fields(cls))
        if hasattr(cls, "__slots__"):
            field_names = [name for name in field_names if name not in cls.__slots__]
        cls_dict["__slots__"] = field_names
        for field_name in field_names:
            cls_dict.pop(field_name, None)

        def __getstate__(self):
            return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

        cls_dict["__getstate__"] = __getstate__

        def __setstate__(self, state):
            for name, value in state.items():
                object.__setattr__(self, name, value)

        cls_dict["__setstate__"] = __setstate__
        if iterable and not hasattr(cls, "__iter__"):

            def __iter__(self):
                for f in dataclasses.fields(self):
                    yield getattr(self, f.name)

            cls_dict["__iter__"] = __iter__
        cls_dict.pop("__dict__", None)
        qualname = getattr(cls, "__qualname__", None)
        cls = type(cls)(cls.__name__, (cls,) + cls.__bases__, cls_dict)
        if qualname is not None:
            cls.__qualname__ = qualname
        return cls

    if cls is None:
        return _decorate

    return _decorate(cls)

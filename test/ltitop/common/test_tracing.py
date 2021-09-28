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

from ltitop.common.tracing import Traceable


class TraceableWithNoPublicAPI(metaclass=Traceable):
    pass


def test_traceable_with_no_public_api():
    obj = TraceableWithNoPublicAPI()
    assert not hasattr(obj, "trace")
    assert not hasattr(obj, "notrace")


class SimpleTraceable(metaclass=Traceable):
    def do_once(self, *args, **kwargs):
        return True

    def do_twice(self, *args, **kwargs):
        with self.notrace():
            ret = self.do_once(*args, **kwargs)
            return ret and self.do_once(*args, **kwargs)


def test_traceable_outside_tracing_scope():
    obj = SimpleTraceable()
    obj.do_once(1, 2, foo="bar")
    obj.do_twice(3.0, True, fizz="buzz")
    assert not obj._traces


def test_traceable_inside_tracing_scope():
    obj = SimpleTraceable()
    with obj.trace() as trace:
        obj.do_once(1, 2, foo="bar")
        obj.do_twice(3.0, True, fizz="buzz")
        assert trace == [
            (SimpleTraceable.do_once, True, (1, 2), {"foo": "bar"}),
            (SimpleTraceable.do_twice, True, (3.0, True), {"fizz": "buzz"}),
        ]

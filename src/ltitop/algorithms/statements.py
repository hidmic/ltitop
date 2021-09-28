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

from typing import Tuple, Union

import sympy

from ltitop.common.dataclasses import immutable_dataclass


@immutable_dataclass
class Statement:
    def perform(self, scope):
        raise NotImplementedError()


@immutable_dataclass
class Assignment(Statement):
    lhs: Union[sympy.Expr, Tuple[sympy.Expr]]
    rhs: Union[sympy.Expr, Tuple[sympy.Expr]]

    __slots__ = "perform"

    def __post_init__(self):
        if not self.lhs or not self.rhs:
            raise ValueError(f"Incomplete assignment: {self}")
        lhs_is_tuple = isinstance(self.lhs, tuple)
        rhs_is_tuple = isinstance(self.rhs, tuple)
        if lhs_is_tuple and rhs_is_tuple:
            if len(self.lhs) != len(self.rhs):
                raise ValueError(f"Unbalanced assignment: {self}")

            def structured_assign(scope=None):
                scope = scope.items() if scope is not None else []
                return {
                    lhs: rhs.subs(scope).doit() for lhs, rhs in zip(self.lhs, self.rhs)
                }

            super().__setattr__("perform", structured_assign)
        elif not lhs_is_tuple and rhs_is_tuple:

            def pack_rvalues(scope=None):
                scope = scope.items() if scope is not None else []
                rvalues = tuple(rhs.subs(scope).doit() for rhs in self.rhs)
                return {self.lhs: sympy.Tuple(*rvalues)}

            super().__setattr__("perform", pack_rvalues)
        elif lhs_is_tuple and not rhs_is_tuple:

            def unpack_rvalue(scope=None):
                scope = scope.items() if scope is not None else []
                rvalue = self.rhs.subs(scope).doit()
                return {self.lhs[i]: rvalue[i] for i in range(len(self.lhs))}

            super().__setattr__("perform", unpack_rvalue)
        else:  # not lhs_is_tuple and not rhs_is_tuple

            def direct_assign(scope=None):
                scope = scope.items() if scope is not None else []
                return {self.lhs: self.rhs.subs(scope).doit()}

            super().__setattr__("perform", direct_assign)

    def __str__(self):
        return "{} = {}".format(
            ", ".join(map(str, self.lhs)) if isinstance(self.lhs, tuple) else self.lhs,
            ", ".join(map(str, self.rhs)) if isinstance(self.rhs, tuple) else self.rhs,
        )

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

from ltitop.algorithms.expressions.arithmetic import NonAssociativeAdd
from ltitop.algorithms.expressions.arithmetic import NonAssociativeMul
from ltitop.algorithms.statements import Assignment


def implementation_hardware(algorithm):
    num_adders = 0
    num_multipliers = 0
    variables = algorithm.inputs + algorithm.states
    for statement in algorithm.procedure:
        if isinstance(statement, Assignment):
            expressions = statement.rhs
            if not isinstance(expressions, tuple):
                expressions = expressions,
            for expr in expressions:
                adders = [
                    atom for atom in expr.atoms(
                        sympy.Add, NonAssociativeAdd
                    ) if atom.has(*variables)
                ]
                multipliers = [
                    atom for atom in expr.atoms(
                        sympy.Mul, NonAssociativeMul
                    ) if atom.has(*variables)
                ]
                num_adders += len(adders)
                num_multipliers += len(multipliers)
    memory_size = 0
    for state in algorithm.states:
        try:
            memory_size += int(sum(state.shape))
        except:
            memory_size += 1
    return num_adders, num_multipliers, memory_size

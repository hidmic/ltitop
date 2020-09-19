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

import networkx as nx

import scipy.signal as signal
import sympy

from ltitop.topology.realizations.direct_forms import DirectFormI
from ltitop.topology.diagram.construction import as_diagram
from ltitop.topology.diagram.construction import series_diagram
from ltitop.topology.diagram.construction import parallel_diagram
from ltitop.topology.diagram.construction import tempvar


def test_tempvar():
    var0, var1, var2, var3 = \
        sympy.symbols('var0 var1 var2 var3')
    diagram = nx.MultiDiGraph()
    assert tempvar(diagram) == var0
    diagram.add_nodes_from([var0, var1, var2])
    assert tempvar(diagram) == var3


def test_block_as_diagram():
    block = DirectFormI.from_model(
        signal.dlti(1, [1, 1, 0.5]))
    x, y = sympy.symbols('x y')
    diagram = as_diagram(
        block, input_=x, output=y
    )
    assert len(diagram[x][y]) == 1
    assert diagram[x][y][0]['block'] is block


def test_series_diagram_construction():
    head_block = DirectFormI.from_model(
        signal.dlti(1, [1, 1, 0.5]))
    tail_block = DirectFormI.from_model(
        signal.dlti(2, [1, -0.5]))
    x, y = sympy.symbols('x y')

    diagram = series_diagram([
        head_block, tail_block
    ], input_=x, output=y)

    paths = list(nx.all_simple_edge_paths(
        diagram, source=x, target=y))
    assert len(paths) == 1
    path = paths[0]
    assert len(path) == 2
    u, v, i = path[0]
    assert u is x
    assert i == 0
    assert diagram[u][v][i]['block'] is head_block

    w, z, i = path[1]
    assert v is w
    assert z is y
    assert i == 0
    assert diagram[w][z][i]['block'] is tail_block


def test_parallel_diagram_construction():
    left_block = DirectFormI.from_model(
        signal.dlti(1, [1, 1, 0.5]))
    right_block = DirectFormI.from_model(
        signal.dlti(2, [1, -0.5]))
    x, y = sympy.symbols('x y')

    diagram = parallel_diagram([
        left_block, right_block
    ], input_=x, output=y)

    assert len(diagram[x][y]) == 2
    assert diagram[x][y][0]['block'] is left_block
    assert diagram[x][y][1]['block'] is right_block

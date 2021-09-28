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

import sys

import pygraphviz as pgv
import sympy


def pretty(diagram):
    agraph = pgv.AGraph(
        directed=True,
        rankdir="TB",
        splines="polyline",
    )
    agraph.add_nodes_from(
        diagram.nodes,
        label="",
        xlabel=r"\N",
        shape="circle",
        color="black",
        style="filled",
        width=0.1,
        fixedsize=True,
    )
    for u, v, w, block in diagram.edges(data="block", keys=True):
        n = f"{v} += f({u}) ({w})"
        agraph.add_node(
            n,
            label=sympy.pretty(block, num_columns=sys.maxsize),
            fontname="courier",
            shape="box",
            style="solid",
        )
        agraph.add_edge(u, n, style="solid", arrowsize=0.5)
        agraph.add_edge(n, v, style="solid", arrowsize=0.5)
    agraph.layout(prog="dot")
    return agraph

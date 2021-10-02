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
import itertools

import networkx as nx
import sympy

from ltitop.models.transforms import to_zpk


def _tempnode(diagram, skip=None, suffix="tmp", factory=str):
    key = suffix, factory
    if key not in diagram.graph:
        node_generator = (factory(f"{suffix}{i}") for i in itertools.count())
        unique_node_generator = (
            node for node in node_generator if node not in diagram.nodes
        )
        diagram.graph[key] = unique_node_generator
    skip = skip or []
    return next(node for node in diagram.graph[key] if node not in skip)


tempvar = functools.partial(_tempnode, suffix="var", factory=sympy.Symbol)


def add_subdiagram(diagram, input_, output, subdiagram):
    mapping = {}

    variables = set(subdiagram.nodes)

    if subdiagram.graph["input"] != input_:
        mapping[subdiagram.graph["input"]] = input_
    variables.remove(subdiagram.graph["input"])

    if subdiagram.graph["output"] != output:
        mapping[subdiagram.graph["output"]] = output
    variables.remove(subdiagram.graph["output"])

    mapping.update(
        {
            variable: tempvar(diagram, skip=subdiagram)
            for variable in variables
            if variable in diagram
        }
    )

    if mapping:
        subdiagram = nx.relabel_nodes(subdiagram, mapping)

    # Manually update the diagram to avoid modifying .graph
    diagram.add_nodes_from(subdiagram.nodes.data())
    diagram.add_edges_from(subdiagram.edges.data())


def input_of(block):
    if not isinstance(block, nx.MultiDiGraph):
        return None
    return block.graph["input"]


def output_of(block):
    if not isinstance(block, nx.MultiDiGraph):
        return None
    return block.graph["output"]


def _unique(var, at):
    if var in at.nodes:
        return None
    return var


def as_diagram(block, input_=None, output=None, **data):
    if input_ and input_ == output:
        raise ValueError(f"{input_} used as input and output")
    if isinstance(block, nx.MultiDiGraph):
        if not input_ and not output:
            return block
        diagram = nx.MultiDiGraph()
        input_ = input_ or input_of(block) or tempvar(diagram)
        output = output or output_of(block) or tempvar(diagram)
        add_subdiagram(diagram, input_, output, block)
    else:
        diagram = nx.MultiDiGraph()
        input_ = input_ or tempvar(diagram)
        output = output or tempvar(diagram)
        diagram.add_edge(input_, output, block=block, **data)
    diagram.graph["input"] = input_
    diagram.graph["output"] = output
    return diagram


def series_diagram(blocks, input_=None, output=None, simplify=True):
    if input_ and input_ == output:
        raise ValueError(f"{input_} used as input and output")
    nonredundant_blocks = []
    for block in blocks:
        if isinstance(block, nx.MultiDiGraph):
            if block.size() > 1:
                nonredundant_blocks.append(block)
                continue
            _, _, block = next(iter(block.edges(data="block")))
        model = to_zpk(block.model)
        if simplify:
            if model.gain == 0.0:
                return as_diagram(block, input_=input_, output=output)
            if model.gain == 1.0 and len(model.poles) == 0 and len(model.zeros) == 0:
                continue
        nonredundant_blocks.append(block)
    if not nonredundant_blocks:
        return as_diagram(blocks[0], input_=input_, output=output)
    diagram = nx.MultiDiGraph()
    diagram.add_node(input_ or input_of(nonredundant_blocks[0]) or tempvar(diagram))
    for i in range(1, len(nonredundant_blocks)):
        diagram.add_node(
            _unique(output_of(nonredundant_blocks[i - 1]), at=diagram)
            or _unique(input_of(nonredundant_blocks[i]), at=diagram)
            or tempvar(diagram)
        )
    diagram.add_node(
        _unique(output, at=diagram)
        or _unique(output_of(nonredundant_blocks[-1]), at=diagram)
        or tempvar(diagram)
    )
    variables = list(diagram.nodes)
    if output and output is not variables[-1]:
        idx = variables.index(output)
        variables[idx] = variables[-1]
        variables[-1] = output
    for (u, v), block in zip(nx.utils.pairwise(variables), nonredundant_blocks):
        if isinstance(block, nx.MultiDiGraph):
            add_subdiagram(diagram, u, v, block)
        else:
            diagram.add_edge(u, v, block=block)
    diagram.graph["input"] = variables[0]
    diagram.graph["output"] = variables[-1]
    return diagram


def parallel_diagram(blocks, input_=None, output=None, simplify=True):
    if input_ and input_ == output:
        raise ValueError(f"{input_} used as input and output")
    nonredundant_blocks = []
    for block in blocks:
        if isinstance(block, nx.MultiDiGraph):
            if block.size() > 1:
                nonredundant_blocks.append(block)
                continue
            _, _, block = next(iter(block.edges(data="block")))
        model = to_zpk(block.model)
        if simplify and model.gain == 0.0:
            continue
        nonredundant_blocks.append(block)
    if not nonredundant_blocks:
        return as_diagram(blocks[0], input_=input_, output=output)
    diagram = nx.MultiDiGraph()
    inputs = (input_of(block) for block in nonredundant_blocks)
    input_ = (
        input_
        or next(
            (
                variable
                for variable in inputs
                if variable is not None and variable not in diagram
            ),
            None,
        )
        or tempvar(diagram)
    )
    outputs = (output_of(block) for block in nonredundant_blocks)
    output = (
        output
        or next(
            (
                variable
                for variable in outputs
                if variable is not None and variable not in diagram
            ),
            None,
        )
        or tempvar(diagram)
    )
    for block in nonredundant_blocks:
        if isinstance(block, nx.MultiDiGraph):
            add_subdiagram(diagram, input_, output, block)
        else:
            diagram.add_edge(input_, output, block=block)
    diagram.graph["input"] = input_
    diagram.graph["output"] = output
    return diagram

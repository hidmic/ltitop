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


def probably(func):
    @functools.wraps(func)
    def __wrapper(*args, probability=1.0):
        if random.random() < probability:
            return func(*args)
        return args
    return __wrapper

@probably
def expand_one_block(diagram):
    u, v, k = edge = random.choice(diagram.edges(keys=True))
    block = diagram.edges[edge]['block']
    diagram.remove_edge(u, v, k)
    if bool(random.getrandbits(1)):  # expand in series
        leading_block, trailing_block = series_decomposition(block)
        w = tmpvar(diagram)
        diagram.add_edge(u, w, block=leading_block)
        diagram.add_edge(w, v, block=trailing_block)
    else:  # expand in parallel
        left_block, right_block = parallel_decomposition(block)
        diagram.add_edge(u, v, block=left_block)
        diagram.add_edge(u, v, block=right_block)
    return diagram,

@probably
def collapse_one_variable(diagram):
    internal_variables = {
        variable for variable, succesors in
        diagram.succ if len(successors) == 1
    }.intersection({
        variable for variable, predecessors in
        diagram.pred if len(predecessors) == 1
    })
    if internal_variables:
        variable = random.choice(internal_variables)
        predecessor = next(diagram.predecessors(variable))
        successor = next(diagram.successors(variable))
        in_edges = diagram.in_edges(variable, keys=True, data='block')
        out_edges = diagram.out_edges(variable, keys=True, data='block')
        block = series_composition([
            parallel_composition([block for *_, block in in_edges]),
            parallel_composition([block for *_, block in out_edges])
        ])
        for u, v, k, _ for itertools.chain(in_edges, out_edges):
            diagram.remove_edge(u, v, k)
        diagram.add_edge(predecessor, successor, block=block)
    return diagram,

@probably
def rebalance_algorithms(diagram):
    for edge in diagram.edges(keys=True):
        block = diagram.edges[edge]['block']
        algorithm = arithmetic.rebalance(block.algorithm)
        block = dataclasses.replace(block, algorithm=algorithm)
        diagram.edges[edge]['block'] = block
    return diagram,

@probably
def multipoint_algorithm_swap(diagram1, diagram2):
    for edge in diagram1.edges(keys=True):
        block1 = diagram1.edges[edge]['block']
        size = len(block1.algorithm.procedure)
        cxpoint = random.randint(0, size)
        if cxpoint == size:
            # Nothing to swap
            continue
        block2 = diagram2.edges[edge]['block']
        if cxpoint != 0:
            algorithm1 = block1.algorithm
            algorithm2 = block2.algorithm
            procedure1 = algorithm1.procedure
            procedure2 = algorithm2.procedure
            procedure1 = procedure1[:cxpoint] + procedure2[cxpoint:]
            procedure2 = procedure1[:cxpoint] + procedure2[cxpoint:]
            algorithm1 = dataclasses.replace(algorithm1, procedure=procedure1)
            algorithm2 = dataclasses.replace(algorithm2, procedure=procedure2)
            block1 = dataclasses.replace(block1, algorithm=algorithm1)
            block2 = dataclasses.replace(block2, algorithm=algorithm2)
        else:
            # Parameters are the same, so simply swap blocks
            block1, block2 = block2, block1
        diagram1.edges[edge]['block'] = block1
        diagram2.edges[edge]['block'] = block2
    return diagram1, diagram2

def optimize(model, implement, evaluate):
    toolbox = deap.base.Toolbox()
    toolbox.register('species', implement, model=model)
    toolbox.register('mate', cxUniformND, ndpb=ndpb)
    toolbox.register('select', deap.tools.selTournament, tournsize=tournsize)
    toolbox.register('mutate', )
    toolbox.register('evaluate', evaluate, model=model)

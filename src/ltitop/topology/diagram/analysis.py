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
import numpy as np

from ltitop.arithmetic.error_bounded import error_bounded
from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.floating_point import mpfloat
from ltitop.arithmetic.interval import interval
from ltitop.common.arrays import asscalar_if_possible
from ltitop.models.analysis import dc_gain as model_dc_gain
from ltitop.models.analysis import is_stable as is_model_stable
from ltitop.models.analysis import output_range as model_output_range
from ltitop.models.analysis import spectral_radius
from ltitop.models.analysis import worst_case_peak_gain as model_worst_case_peak_gain


def is_stable(diagram, source=None, target=None):
    if source or target:
        source = source or diagram.graph["input"]
        target = target or diagram.graph["output"]
        if source not in diagram:
            raise ValueError(f"{source} not in diagram")
        if target not in diagram:
            raise ValueError(f"{target} not in diagram")
        diagram = signal_path(diagram, source, target)
    models = (block.model for _, _, block in diagram.edges(data="block"))
    return all(is_model_stable(model) for model in models)


def spectral_radii(diagram, source=None, target=None):
    if source or target:
        source = source or diagram.graph["input"]
        target = target or diagram.graph["output"]
        if source not in diagram:
            raise ValueError(f"{source} not in diagram")
        if target not in diagram:
            raise ValueError(f"{target} not in diagram")
        diagram = signal_path(diagram, source, target)
    models = (block.model for _, _, block in diagram.edges(data="block"))
    radii = [spectral_radius(model) for model in models]
    return [r for r in radii if r is not None]


def analytic_diagram(diagram, source_ranges=None):
    assert not any(nx.simple_cycles(diagram))

    if source_ranges:
        if not isinstance(source_ranges, dict):
            variable_ranges = {diagram.graph["input"]: source_ranges}
        else:
            variable_ranges = dict(source_ranges)
    else:
        variable_ranges = None

    analytic_diagram = nx.DiGraph()
    analytic_diagram.graph["input"] = diagram.graph["input"]
    analytic_diagram.graph["output"] = diagram.graph["output"]
    for variable in nx.topological_sort(diagram):
        dependency_ranges = [] if variable_ranges else None
        for dependency in diagram.predecessors(variable):
            if variable_ranges:
                if dependency not in variable_ranges:
                    raise ValueError(f"{dependency} cannot be computed")
                input_range = variable_ranges[dependency]
            dc_gain = 0
            worst_case_peak_gain = 0
            error_bounds = interval(0)
            for data in diagram[dependency][variable].values():
                block = data["block"]
                dc_gain += model_dc_gain(block.model)
                worst_case_peak_gain += model_worst_case_peak_gain(block.model)
                if variable_ranges:
                    error_bounds += model_output_range(
                        block.computation_error_model,
                        block.computation_error_bounds(input_range),
                    )
                    dependency_ranges.append(
                        model_output_range(block.model, mpfloat(input_range))
                    )

            analytic_diagram.add_edge(
                dependency,
                variable,
                dc_gain=dc_gain,
                worst_case_peak_gain=worst_case_peak_gain,
                error_bounds=error_bounds,
            )

            for transitive_dependency in analytic_diagram.predecessors(dependency):
                data = analytic_diagram[transitive_dependency][dependency]

                if not np.isscalar(dc_gain):
                    transitive_dc_gain = dc_gain @ data["dc_gain"]
                else:
                    transitive_dc_gain = dc_gain * data["dc_gain"]

                if not np.isscalar(worst_case_peak_gain):
                    transitive_worst_case_peak_gain = (
                        worst_case_peak_gain @ data["worst_case_peak_gain"]
                    )
                else:
                    transitive_worst_case_peak_gain = (
                        worst_case_peak_gain * data["worst_case_peak_gain"]
                    )

                transitive_error_bounds = output_range(
                    analytic_diagram, data["error_bounds"], dependency, variable
                )

                if variable not in analytic_diagram[transitive_dependency]:
                    analytic_diagram.add_edge(
                        transitive_dependency,
                        variable,
                        dc_gain=0,
                        worst_case_peak_gain=0,
                        error_bounds=interval(0),
                    )
                data = analytic_diagram[transitive_dependency][variable]
                data["dc_gain"] += transitive_dc_gain
                data["worst_case_peak_gain"] += transitive_worst_case_peak_gain
                data["error_bounds"] += transitive_error_bounds
        if dependency_ranges:
            error_bounds = np.sum(
                [error_bounded(fixed(r)) for r in dependency_ranges]
            ).error_bounds
            for edge in analytic_diagram.in_edges(variable):
                analytic_diagram.edges[edge]["error_bounds"] += error_bounds
            variable_ranges[variable] = np.sum(dependency_ranges)
    return analytic_diagram


def dc_gain(diagram, source=None, target=None):
    source = source or diagram.graph["input"]
    target = target or diagram.graph["output"]
    if source not in diagram:
        raise ValueError(f"{source} not in diagram")
    if target not in diagram:
        raise ValueError(f"{target} not in diagram")
    if target not in diagram[source]:
        raise ValueError(f"No path from {source} to {target}")
    return diagram[source][target]["dc_gain"]


def worst_case_peak_gain(diagram, source=None, target=None):
    source = source or diagram.graph["input"]
    target = target or diagram.graph["output"]
    if source not in diagram:
        raise ValueError(f"{source} not in diagram")
    if target not in diagram:
        raise ValueError(f"{target} not in diagram")
    if target not in diagram[source]:
        raise ValueError(f"No path from {source} to {target}")
    return diagram[source][target]["worst_case_peak_gain"]


def error_bounds(diagram, source=None, target=None):
    source = source or diagram.graph["input"]
    target = target or diagram.graph["output"]
    if source not in diagram:
        raise ValueError(f"{source} not in diagram")
    if target not in diagram:
        raise ValueError(f"{target} not in diagram")
    if target not in diagram[source]:
        raise ValueError(f"No path from {source} to {target}")
    return diagram[source][target]["error_bounds"]


def output_range(diagram, source_range, source=None, target=None):
    source_mean = (source_range.upper_bound + source_range.lower_bound) / 2

    gain = dc_gain(diagram, source, target)
    if not np.isscalar(gain):
        target_mean = gain @ source_mean
    else:
        target_mean = gain * source_mean

    source_delta = (source_range.upper_bound - source_range.lower_bound) / 2

    gain = worst_case_peak_gain(diagram, source, target)
    if not np.isscalar(gain):
        target_delta = gain @ source_delta
    else:
        target_delta = gain * source_delta

    target_mean = asscalar_if_possible(target_mean)
    target_delta = asscalar_if_possible(target_delta)

    a = target_mean - target_delta
    b = target_mean + target_delta
    return interval(np.min([a, b], axis=0), np.max([a, b], axis=0))


def signal_path(diagram, source=None, target=None):
    source = source or diagram.graph["input"]
    target = target or diagram.graph["output"]
    if source not in diagram:
        raise ValueError(f"{source} not in diagram")
    if target not in diagram:
        raise ValueError(f"{target} not in diagram")

    paths = nx.all_simple_paths(diagram, source, target)
    nodes = {node for path in paths for node in path}
    return diagram.subgraph(nodes)


def signal_processing_function(diagram, source=None, target=None):
    # TODO(hidmic): support multiple sources
    source = source or diagram.graph["input"]
    target = target or diagram.graph["output"]
    if source not in diagram:
        raise ValueError(f"{source} not in diagram")
    if target not in diagram:
        raise ValueError(f"{target} not in diagram")
    assert not any(nx.simple_cycles(diagram))

    diagram = signal_path(diagram, source, target)

    functional_relations = {}
    topologically_sorted_dependency_graph = []
    for variable in nx.topological_sort(diagram):
        if variable is source:
            continue
        dependency_list = tuple(diagram.predecessors(variable))
        for dependency in dependency_list:
            functional_relations[variable, dependency] = tuple(
                data["block"].to_function()
                for data in diagram[dependency][variable].values()
            )
        topologically_sorted_dependency_graph.append((variable, dependency_list))

    def _function(inputs):
        values = {source: inputs}
        for variable, dependencies in topologically_sorted_dependency_graph:
            terms = [
                func(values[dependency])
                for dependency in dependencies
                for func in functional_relations[variable, dependency]
            ]
            assert terms  # TODO(hidmic): support source passivation
            values[variable] = np.sum(terms, axis=0) if len(terms) > 1 else terms[0]
        return values[target]

    return _function

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
import functools
import hashlib
import math
import operator

import deap.algorithms
import deap.base
import deap.gp
import deap.tools
import numpy as np

from ltitop.common.functional import argtransform
from ltitop.common.helpers import methodcall
from ltitop.models.composites import parallel_decomposition, series_decomposition
from ltitop.topology.diagram.construction import (
    as_diagram,
    parallel_diagram,
    series_diagram,
)


def md5(*items):
    return hashlib.md5(repr(items).encode()).hexdigest()[:15]


def _series(*operators, variant, tol):
    def __implementation(model, **kwargs):
        return series_diagram(
            [
                operator(submodel)
                for operator, submodel in zip(
                    operators,
                    series_decomposition(model, len(operators), variant, tol=tol),
                )
            ],
            **kwargs,
        )

    return __implementation


def _parallel(*operators, variant, tol):
    def __implementation(model, **kwargs):
        return parallel_diagram(
            [
                operator(submodel)
                for operator, submodel in zip(
                    operators,
                    parallel_decomposition(model, len(operators), variant, tol=tol),
                )
            ],
            **kwargs,
        )

    return __implementation


def _realize(model, *, form, **kwargs):
    return as_diagram(form.from_model(model, **kwargs))


def _evaluate(code, *, func, model, compiler):
    fitness = func(compiler(code), model)
    if dataclasses.is_dataclass(fitness):
        fitness = dataclasses.astuple(fitness)
    return fitness


def _graph(code, pset):
    nodes, edges, labels = deap.gp.graph(code)
    labels = {i: label_for(pset.context[name]) for i, name in labels.items()}
    return nodes, edges, labels


class Code(deap.gp.PrimitiveTree):
    class Fitness(deap.base.Fitness):
        def __init__(self, weights):
            self.weights = weights
            super().__init__()

        def __deepcopy__(self, memo):
            copy = self.__class__(self.weights)
            copy.wvalues = self.wvalues
            memo[id(self)] = copy
            return copy

    def __init__(self, content, weights=None):
        super().__init__(content)
        if weights is not None:
            self.fitness = Code.Fitness(weights)


def formulate(
    prototype, *, transforms, evaluate, weights, forms, variants, dtype=float, tol=1e-16
):
    pfunc = deap.base.Toolbox()
    pfunc.register("series", _series)
    pfunc.register("parallel", _parallel)
    pfunc.register("realize", argtransform(_realize, *transforms))

    pset = deap.gp.PrimitiveSet("factory", 0)
    for variant in variants:
        suffix = md5(variant, tol)
        primitive = functools.partial(
            methodcall(pfunc, "series"), variant=variant, tol=tol
        )
        pset.addPrimitive(primitive, 2, name=f"series{suffix}")
        primitive = functools.partial(
            methodcall(pfunc, "parallel"), variant=variant, tol=tol
        )
        pset.addPrimitive(primitive, 2, name=f"parallel{suffix}")
        for form in forms:
            primitive = functools.partial(
                methodcall(pfunc, "realize"), form=form, variant=variant, dtype=dtype
            )
            suffix = md5(form.__name__, variant, dtype.__name__)
            pset.addTerminal(primitive, name=f"realize{suffix}")

    toolbox = deap.base.Toolbox()
    toolbox.pset = pset
    toolbox.pfunc = pfunc
    order = len(prototype.poles)
    toolbox.register(
        "code",
        deap.gp.genHalfAndHalf,
        pset=pset,
        min_=1,
        max_=max(math.ceil(math.log2(order)), 1),
    )
    toolbox.register("code_snippet", deap.gp.genFull, pset=pset, min_=0, max_=2)
    toolbox.register("individual", lambda: Code(toolbox.code(), weights))
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", deap.gp.compile, pset=pset)
    toolbox.register(
        "evaluate",
        functools.partial(
            _evaluate, func=evaluate, model=prototype, compiler=toolbox.compile
        ),
    )
    toolbox.register("graph", _graph, pset=pset)

    order_limit = deap.gp.staticLimit(
        key=operator.attrgetter("height"), max_value=math.ceil(math.log2(order))
    )
    toolbox.register("mutate", deap.gp.mutUniform, expr=toolbox.code_snippet, pset=pset)
    toolbox.decorate("mutate", order_limit)
    toolbox.register("mate", deap.gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.decorate("mate", order_limit)

    return toolbox


def label_for(value):
    if hasattr(value, "__name__"):
        return value.__name__
    if isinstance(value, functools.partial):
        return "{}[{}]".format(
            label_for(value.func),
            "; ".join(
                f"{name} = {label_for(value)}" for name, value in value.keywords.items()
            ),
        )
    if isinstance(value, methodcall):
        return value.method_name
    return str(value)


def nsga2(
    population,
    toolbox,
    *,
    mu=None,
    lambda_=None,
    cxpb=0.3,
    mutpb=0.05,
    ngen=100,
    stats=None,
    halloffame=None,
    inf=1e200,
    verbose=__debug__,
):
    logbook = deap.tools.Logbook()
    logbook.header = ["gen", "nevals", "nunfeas"]
    if stats is not None:
        logbook.header += stats.fields

    mu = mu or len(population)
    lambda_ = lambda_ or len(population)

    # Evaluate the individuals with an invalid fitness
    nunfeas = 0
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        if fit is None:
            population.remove(ind)
            nunfeas += 1
            continue
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # This is just to assign the crowding distance to the individuals
    population = deap.tools.selNSGA2(population, len(population))

    nevals = len(invalid_ind) - nunfeas
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, nunfeas=nunfeas, **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = deap.tools.selTournamentDCD(population, lambda_)

        offspring = deap.algorithms.varOr(
            offspring, toolbox, len(offspring), cxpb, mutpb
        )

        # Evaluate the individuals with an invalid fitness
        nunfeas = 0
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            if fit is None:
                offspring.remove(ind)
                nunfeas += 1
                continue
            ind.fitness.values = fit

        population = deap.tools.selNSGA2(population + offspring, mu, nd="log")

        if halloffame is not None:
            halloffame.update(population)

        nevals = len(invalid_ind) - nunfeas
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, nunfeas=nunfeas, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def argnondominated(*scores):
    dominated = []
    scores = np.array(scores, dtype=object).T
    for i in range(len(scores) - 1):
        if i in dominated:
            continue
        for j in range(i + 1, len(scores)):
            if all(scores[i] >= scores[j]):
                dominated.append(j)
            elif all(scores[j] >= scores[i]):
                dominated.append(i)
                break
    return [k for k in range(len(scores)) if k not in dominated]

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
import pickle
import random
import multiprocessing

import numpy as np
np.set_printoptions(linewidth=1000)
import scipy.signal as signal

import deap.algorithms
import deap.gp
import deap.tools

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.24, 7.68)
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.mplot3d

from ltitop.arithmetic.errors import OverflowError
from ltitop.arithmetic.errors import UnderflowError
from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.modular import wraparound
from ltitop.arithmetic.rounding import nearest_integer
from ltitop.arithmetic.fixed_point.processing_unit import ProcessingUnit
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit \
    import FixedFormatArithmeticLogicUnit
from ltitop.arithmetic.fixed_point import fixed
from ltitop.arithmetic.fixed_point.formats import Q

from ltitop.algorithms.analysis import implementation_hardware

from ltitop.models.transforms import discretize
from ltitop.models.transforms import lowpass_to_lowpass

from ltitop.topology.diagram.analysis import spectral_radii
from ltitop.topology.diagram.analysis import output_range
from ltitop.topology.diagram.analysis import error_bounds
from ltitop.topology.diagram.analysis import signal_processing_function
from ltitop.topology.diagram.analysis import analytic_diagram
from ltitop.topology.diagram.construction import series_diagram
from ltitop.topology.diagram.visualization import pretty
from ltitop.topology.realizations.direct_forms import DirectFormI
from ltitop.topology.realizations.direct_forms import DirectFormII

import ltitop.solvers

inf = 1e200  # GP solver does not handle true infinity

def evaluate(
    implement, prototype, *,
    input_noise,
    input_noise_power_density,
    psd, expected_response
):
    # Implement filter
    try:
        diagram = implement(prototype)
    except (OverflowError, UnderflowError, ValueError):
        # simply unfeasible e.g. due to numerical errors
        return None  # unfeasible, discard

    func = signal_processing_function(diagram)
    output_noise = func(input_noise).T[0].astype(float)
    _, output_noise_power_density = psd(output_noise)
    response = 10 * np.log10(
        output_noise_power_density /
        input_noise_power_density
        + 1/inf)  # ensure nonzero values
    return np.sqrt(np.mean((expected_response - response)**2)),

def main():
    random.seed(7)
    np.random.seed(7)

    fs = 200  # Hz
    fo = 40  # Hz
    rp = 1   # dB
    rs = 80  # dB
    fpass = fo - 5  # Hz
    fstop = fo + 5  # Hz
    # N, fo = signal.cheb1ord(fpass, fstop, rp, rs, fs=fs)
    # N, fo = signal.cheb2ord(fpass, fstop, rp, rs, fs=fs)
    # N, fo = signal.buttord(fpass, fstop, rp, rs, fs=fs)
    N, fo = signal.ellipord(fpass, fstop, rp, rs, fs=fs)
    wp = 2 * fs * np.tan(np.pi * fo / fs)
    # prototype = signal.lti(*signal.cheb1ap(N, rp))
    # prototype = signal.lti(*signal.cheb2ap(N, rs))
    # prototype = signal.lti(*signal.buttap(N))
    prototype = signal.lti(*signal.ellipap(N=N, rp=rp, rs=rs))
    model = lowpass_to_lowpass(prototype, wo=wp)
    model = discretize(model, dt=1/fs)

    # Use PSD output to white noise input PSD ratio as response
    window = signal.get_window('blackman', 256)
    psd = functools.partial(
        signal.welch, scaling='density', window=window, fs=fs)
    input_range = interval(lower_bound=-0.5, upper_bound=0.5)
    input_noise_power_density = 0.0005
    input_noise = np.random.normal(
        scale=np.sqrt(input_noise_power_density * fs / 2), size=512)
    assert np.max(input_noise) < ProcessingUnit.active().rinfo().max
    assert np.min(input_noise) > ProcessingUnit.active().rinfo().min
    input_noise = np.array([fixed(n) for n in input_noise])
    _, outputs = model.output(input_noise.astype(float), t=None)
    output_noise = outputs.T[0]
    freq, output_noise_power_density = psd(output_noise)
    expected_response = 10 * np.log10(
        output_noise_power_density /
        input_noise_power_density + 1/inf
    )
    # Take quantization noise into account
    noise_floor = -6.02 * (ProcessingUnit.active().wordlength - 1) - 1.76
    expected_response = np.maximum(expected_response, noise_floor)

    # Formulate GP problem
    toolbox = ltitop.solvers.gp.formulate(
        prototype,
        transforms=[
            functools.partial(lowpass_to_lowpass, wo=wp),
            functools.partial(discretize, dt=1/fs)],
        evaluate=functools.partial(
            evaluate,
            input_noise=input_noise,
            input_noise_power_density=input_noise_power_density,
            psd=psd, expected_response=expected_response
        ),
        weights=(-1.,),
        forms=[DirectFormI, DirectFormII],
        variants=range(1000),
        dtype=fixed,
        tol=1e-6
    )
    toolbox.register('select', deap.tools.selTournament, tournsize=3)

    # Solve GP problem
    only_visualize = True
    if not only_visualize:
        with multiprocessing.Pool() as pool:
            try:
                toolbox.register('map', pool.map)

                stats = deap.tools.Statistics(
                    key=lambda code: code.fitness.values)
                stats.register('avg', np.mean)
                stats.register('med', np.median)
                stats.register('max', np.max)
                stats.register('min', np.min)
                halloffame = deap.tools.HallOfFame(3)
                population = toolbox.population(512)
                population, logbook = deap.algorithms.eaSimple(
                    population, toolbox, cxpb=0.5, mutpb=0.05,
                    ngen=25, stats=stats, halloffame=halloffame,
                    verbose=True)
            finally:
                toolbox.register('map', map)

        with open('halloffame.pkl', 'wb') as f:
            pickle.dump(halloffame, f)
        with open('logbook.pkl', 'wb') as f:
            pickle.dump(logbook, f)
    else:
        with open('halloffame.pkl', 'rb') as f:
            halloffame = pickle.load(f)
        with open('logbook.pkl', 'rb') as f:
            logbook = pickle.load(f)

    plt.figure()
    plt.plot(*logbook.select('gen', 'med'))
    plt.ylabel(r'Response error $Median[\sqrt{E^2(f)}]$ [dBr]')
    plt.xlabel('Generations')
    plt.legend(loc='upper right')
    plt.savefig('evolution.png')

    plt.figure()
    plt.plot(*logbook.select('gen', 'nunfeas'))
    plt.ylabel('Unfeasibles')
    plt.xlabel('Generations')
    plt.savefig('unfeasibles.png')

    implement = toolbox.compile(halloffame[0])
    optimized_implementation = implement(prototype)

    pretty(optimized_implementation).draw('optimized.png')

    func = signal_processing_function(optimized_implementation)
    output_noise = func(input_noise).T[0].astype(float)
    _, output_noise_power_density = psd(output_noise)
    optimized_implementation_response = \
        10 * np.log10(output_noise_power_density / input_noise_power_density + 1/inf)

    show_biquad_cascade = True
    if show_biquad_cascade:
        biquad_cascade = series_diagram([
            DirectFormI.from_model(
                signal.dlti(section[:3], section[3:], dt=model.dt), dtype=fixed
            ) for section in signal.zpk2sos(model.zeros, model.poles, model.gain)
        ], simplify=False)

        pretty(biquad_cascade).draw('biquad.png')
        func = signal_processing_function(biquad_cascade)
        output_noise = func(input_noise).T[0].astype(float)
        _, output_noise_power_density = psd(output_noise)
        biquad_cascade_response = \
            10 * np.log10(output_noise_power_density / input_noise_power_density + 1/inf)

    plt.figure()
    plt.plot(freq, expected_response, label='Model')
    if show_biquad_cascade:
        plt.plot(freq, biquad_cascade_response, label='Biquad cascade')
    plt.plot(freq, optimized_implementation_response, label='Optimized realization')
    plt.xlabel('Frecuency $f$ [Hz]')
    plt.ylabel('Response $|H(f)|$ [dBr]')
    plt.legend()
    plt.savefig('response.png')
    plt.show()


if __name__ == '__main__':
    with FixedFormatArithmeticLogicUnit(
        format_=Q(15), allows_overflow=True,
        overflow_behavior=wraparound
    ):
        main()

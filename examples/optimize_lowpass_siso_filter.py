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

import ltitop.common.functional as functional

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

import ltitop.solvers as solvers


inf = 1e200  # GP solver does not handle true infinity

@dataclasses.dataclass
class Criteria:
    overflow_margin : float = -inf
    underflow_margin : float = -inf
    stability_margin : float = -inf
    arithmetic_snr : float = -inf
    frequency_response_error : float = inf
    number_of_adders : int = inf
    number_of_multipliers : int = inf
    size_of_memory : int = inf


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
    print(model)

    # Use PSD output to white noise input PSD ratio as response
    window = signal.get_window('blackman', 256)
    input_range = interval(lower_bound=-0.5, upper_bound=0.5)
    input_noise_power_density = 0.0005
    input_noise = np.random.normal(
        scale=np.sqrt(input_noise_power_density * fs / 2), size=512)
    assert np.max(input_noise) < ProcessingUnit.active().rinfo().max
    assert np.min(input_noise) < ProcessingUnit.active().rinfo().min
    input_noise = np.array([fixed(n) for n in input_noise])
    _, outputs = model.output(input_noise.astype(float), t=None)
    output_noise = outputs.T[0]
    freq, output_noise_power_density = signal.welch(
        output_noise, fs, window=window)
    expected_response = 10 * np.log10(
        output_noise_power_density /
        input_noise_power_density + 1/inf
    )
    # Take quantization noise into account
    noise_floor = -6.02 * (ProcessingUnit.active().wordlength - 1) - 1.76
    expected_response = np.maximum(expected_response, noise_floor)

    # Formulate GP problem
    toolbox = solvers.gp.formulate(
        order=len(model.poles),
        weights=(1., 1., 1., 1., -1., -1., -1., -1.),
        forms=[DirectFormI, DirectFormII],
        variants=range(1000),
        dtype=fixed,
        tol=1e-6
    )

    toolbox.pfunc.decorate(
        'realize',  # model from prototype
        functional.atransform(
            functools.partial(lowpass_to_lowpass, wo=wp),
            functools.partial(discretize, dt=1/fs)
        )
    )

    def evaluate(code):
        criteria = Criteria()

        implement = toolbox.compile(code)

        # Implement filter
        try:
            diagram = implement(prototype)
        except OverflowError as e:
            criteria.overflow_margin = float(np.min(e.margin))
            return criteria
        except UnderflowError as e:
            criteria.underflow_margin = float(np.min(e.margin))
            return criteria
        except ValueError as e:
            # simply unfeasible e.g. due to numerical errors
            return None

        # Evaluate filter stability
        radii = spectral_radii(diagram)
        criteria.stability_margin = inf
        if radii:
            max_spectral_radius = np.max(radii)
            if max_spectral_radius != 0:
                criteria.stability_margin = -np.log10(max_spectral_radius)
            if criteria.stability_margin < 1e-5:
                # Not enough margin to proceed
                return criteria

        # Compute variable and error bounds
        try:
            adiagram = analytic_diagram(diagram, input_range)
        except OverflowError as e:
            criteria.overflow_margin = float(np.min(e.margin))
            return criteria
        except UnderflowError as e:
            criteria.underflow_margin = float(np.min(e.margin))
            return criteria

        # Assume zero margin
        criteria.overflow_margin = 0
        criteria.underflow_margin = 0

        def power(signal_range):
            # Assume uniform distribution
            a = signal_range.lower_bound
            b = signal_range.upper_bound
            return (a**2 + a * b + b**2) / 3  # E(signal^2)

        signal_power = power(output_range(adiagram, input_range))
        if signal_power != 0.:
            # Estimate SNR due to arithmetic 'noise'
            arithmetic_noise_power = power(error_bounds(adiagram))
            criteria.arithmetic_snr = 10. * np.log10(float(
                signal_power / arithmetic_noise_power
            )).item()  # force it to be an scalar

            # Compute frequency response error
            func = signal_processing_function(diagram)
            output_noise = func(input_noise).T[0]
            _, output_noise_power_density = signal.welch(
                output_noise.astype(float), fs,
                window=window)
            response = 10 * np.log10(
                output_noise_power_density /
                input_noise_power_density
                + 1/inf)  # ensure nonzero values
            criteria.frequency_response_error = \
                np.sqrt(np.mean((expected_response - response)**2))
                # np.max(np.abs(expected_response - response))

        # Summarize hardware
        criteria.number_of_adders, \
        criteria.number_of_multipliers, \
        criteria.size_of_memory = np.sum([
            implementation_hardware(block.algorithm)
            for _, _, block in diagram.edges(data='block')
        ], axis=0)
        criteria.number_of_adders += diagram.number_of_edges() - 1

        return criteria

    toolbox.register('evaluate', evaluate)

    # Solve GP problem
    only_visualize = True
    if not only_visualize:
        stats = deap.tools.Statistics(
            key=lambda code: code.fitness.values)
        stats.register('avg', np.mean, axis=0)
        stats.register('med', np.median, axis=0)
        stats.register('min', np.min, axis=0)
        pareto_front = deap.tools.ParetoFront()
        population = toolbox.population(256)
        population, logbook = solvers.gp.nsga2(
            population, toolbox, mu=256, lambda_=64,
            cxpb=0.5, mutpb=0.2, ngen=25, stats=stats,
            halloffame=pareto_front, verbose=True)

        with open('front.pkl', 'wb') as f:
            pickle.dump(pareto_front, f)
        with open('logbook.pkl', 'wb') as f:
            pickle.dump(logbook, f)
    else:
        with open('front.pkl', 'rb') as f:
            pareto_front = pickle.load(f)
        with open('logbook.pkl', 'rb') as f:
            logbook = pickle.load(f)
    codes = []
    criteria = []
    for code in pareto_front:
        crt = Criteria(*code.fitness.values)
        if crt.frequency_response_error >= inf:
            continue
        if crt.stability_margin <= 0:
            continue
        if crt.overflow_margin < 0:
            continue
        if crt.underflow_margin < 0:
            continue
        codes.append(code)
        criteria.append(crt)

    frequency_response_error = np.array([
        crt.frequency_response_error for crt in criteria])
    arithmetic_snr = np.array([crt.arithmetic_snr for crt in criteria])
    stability_margin = np.array([crt.stability_margin for crt in criteria])
    memory_size = np.array([crt.size_of_memory for crt in criteria])
    memory_size_in_bytes = memory_size * ProcessingUnit.active().wordlength / 8

    plt.figure()
    gen, med = logbook.select('gen', 'med')
    crt = Criteria(*np.array(med).T)

    def fit_most_within_unit_interval(values):
        values = np.asarray(values)
        med = np.median(values)
        mad = np.median(np.absolute(values - med))
        if not mad:
            return values
        return (values - med) / (4. * mad)

    plt.plot(gen, fit_most_within_unit_interval(crt.overflow_margin), label='Med[$M_o$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.underflow_margin), label='Med[$M_u$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.stability_margin), label='Med[$M_s$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.arithmetic_snr), label='Med[$SNR_{arit}$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.frequency_response_error), label='Med[$E_2$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.number_of_adders), label='Med[$N_a$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.number_of_multipliers), label='Med[$N_m$]')
    plt.plot(gen, fit_most_within_unit_interval(crt.size_of_memory), label='Med[$N_e$]')
    plt.ylabel('Normalized evolution')
    plt.xlabel('Generations')
    plt.ylim([-10, 10])
    plt.legend(loc='upper right')
    plt.savefig('population_evolution.png')

    plt.figure()
    plt.plot(*logbook.select('gen', 'nunfeas'))
    plt.ylabel('Unfeasibles')
    plt.xlabel('Generations')
    plt.savefig('unfeasibles.png')

    def pareto2d(ax, x, y):
        ax.scatter(
            x, y, marker='o',
            facecolors='none',
            edgecolors='r')
        idx = np.argsort(x)
        ax.plot(x[idx], y[idx], 'k--')

    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    idx = solvers.gp.argnondominated(-frequency_response_error, stability_margin)
    pareto2d(ax, frequency_response_error[idx], stability_margin[idx])
    ax.set_ylabel('Margin $M_s$')
    ax.invert_xaxis()
    ax = fig.add_subplot(3, 1, 2)
    idx = solvers.gp.argnondominated(-frequency_response_error, arithmetic_snr)
    pareto2d(ax, frequency_response_error[idx], arithmetic_snr[idx])
    ax.set_ylabel('$SNR_{arit}$ [dBr]')
    ax.invert_xaxis()
    ax = fig.add_subplot(3, 1, 3)
    idx = solvers.gp.argnondominated(-frequency_response_error, -memory_size_in_bytes)
    pareto2d(ax, frequency_response_error[idx], memory_size_in_bytes[idx])
    ax.set_xlabel('Error $E_2$ [dBr]')
    ax.set_ylabel('Total $N_e$ @ Q15 [bytes]')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.savefig('pareto_fronts.png')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    idx = solvers.gp.argnondominated(
        -frequency_response_error,
        arithmetic_snr,
        -memory_size_in_bytes)
    scatter = ax.scatter(
        frequency_response_error[idx],
        arithmetic_snr[idx],
        memory_size_in_bytes[idx],
        c=stability_margin[idx],
        cmap='jet_r'
    )

    fig.colorbar(scatter, ax=ax, label='Margin $M_s$')
    ax.set_xlabel('Error $E_2$ [dBr]')
    ax.set_ylabel('$SNR_{arit}$ [dBr]')
    ax.set_zlabel('Total $N_e$ @ Q15 [bytes]')
    ax.invert_xaxis()
    ax.invert_zaxis()
    plt.savefig('pareto_sampling.png')

    index = np.argsort(frequency_response_error)[0]
    implement = toolbox.compile(codes[index])
    optimized_implementation_a = implement(prototype)
    index = np.argsort(frequency_response_error)[3]
    implement = toolbox.compile(codes[index])
    optimized_implementation_b = implement(prototype)

    print(memory_size_in_bytes[np.argsort(frequency_response_error)])
    print(stability_margin[np.argsort(frequency_response_error)])
    print(arithmetic_snr[np.argsort(frequency_response_error)])

    pretty(optimized_implementation_a).draw('optimized_a.png')
    pretty(optimized_implementation_b).draw('optimized_b.png')

    func = signal_processing_function(optimized_implementation_a)
    output_noise = func(input_noise).T[0]
    _, output_noise_power_density = signal.welch(
        output_noise.astype(float), fs,
        window=window, scaling='density')
    optimized_implementation_a_response = \
        10 * np.log10(output_noise_power_density / input_noise_power_density + 1/inf)

    func = signal_processing_function(optimized_implementation_b)
    output_noise = func(input_noise).T[0]
    _, output_noise_power_density = signal.welch(
        output_noise.astype(float), fs,
        window=window, scaling='density')
    optimized_implementation_b_response = \
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
        output_noise = func(input_noise).T[0]
        _, output_noise_power_density = signal.welch(
            output_noise.astype(float), fs, window=window)
        biquad_cascade_response = \
            10 * np.log10(output_noise_power_density / input_noise_power_density + 1/inf)

    plt.figure()
    plt.plot(freq, expected_response, label='Model')
    if show_biquad_cascade:
        plt.plot(freq, biquad_cascade_response, label='Biquad cascade')
        pass
    plt.plot(freq, optimized_implementation_a_response, label='Optimized realization A')
    plt.plot(freq, optimized_implementation_b_response, label='Optimized realization B')
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

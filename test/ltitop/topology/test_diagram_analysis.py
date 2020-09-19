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

import scipy.signal as signal
import sympy

from ltitop.arithmetic.interval import interval
from ltitop.arithmetic.fixed_point.formats import uQ, Q
from ltitop.arithmetic.fixed_point.fixed_format_arithmetic_logic_unit \
    import FixedFormatArithmeticLogicUnit
from ltitop.arithmetic.rounding import nearest_integer

from ltitop.models.composites import parallel_composition
from ltitop.models.composites import series_composition

from ltitop.topology.realizations.direct_forms import DirectFormI
from ltitop.topology.diagram.construction import series_diagram
from ltitop.topology.diagram.construction import parallel_diagram

from ltitop.topology.diagram.analysis import is_stable
from ltitop.topology.diagram.analysis import dc_gain
from ltitop.topology.diagram.analysis import worst_case_peak_gain
from ltitop.topology.diagram.analysis import output_range
from ltitop.topology.diagram.analysis import error_bounds
from ltitop.topology.diagram.analysis import signal_processing_function
from ltitop.topology.diagram.analysis import analytic_diagram

from numpy.testing import assert_allclose


def test_diagram_is_stable():
    x, y, z = sympy.symbols('x y z')
    diagram = series_diagram([
        parallel_diagram([
            DirectFormI.from_model(
                # F(z^-1) = z^-1 / (1 + z^-1)
                # UNSTABLE
                signal.dlti([1], [1, 1])
            ),
            DirectFormI.from_model(
                # F(z) = z^-1 / (1 + z^-1 + 0.5 z^-2)
                signal.dlti([1, 0], [1, 1, 0.5])
            )
        ], output=y),
        DirectFormI.from_model(
            # F(z) = z^-1 / (1 - 0.1 z^-1)
            signal.dlti([1], [1, -0.1])
        )
    ], input_=x, output=z)

    assert not is_stable(diagram)
    assert not is_stable(diagram, source=x, target=y)
    assert is_stable(diagram, source=y, target=z)


def test_diagram_dc_gain():
    x, y, z = sympy.symbols('x y z')
    diagram = series_diagram([
        parallel_diagram([
            DirectFormI.from_model(
                # F(z^-1) = 2 / (1 - 0.5 z^-1)
                # DC GAIN = F(z = 1)
                # DC GAIN = 2 / (1 - 0.5) = 4
                signal.dlti([2, 0], [1, -0.5])
            ),
            DirectFormI.from_model(
                # F(z^-1) = 1 / (1 + z^-1 + 0.5 z^-2)
                # DC GAIN = F(z = 1)
                # DC GAIN = 1 / (1 + 1 + 0.5) = 1 / 2.5
                signal.dlti([1, 0, 0], [1, 1, 0.5])
            ),
        ], output=y),
        DirectFormI.from_model(
            # F(z^-1) = 1 / (1 - 0.1 z^-1)
            # DC GAIN = F(z = 1)
            # DC GAIN = 1 / (1 - 0.1) = 1 / 0.9
            signal.dlti([1, 0], [1, -0.1])
        )
    ], input_=x, output=z)

    analytical_diagram = analytic_diagram(diagram)
    assert dc_gain(analytical_diagram) == (4.0 + 1.0 / 2.5) * 1.0 / 0.9
    assert dc_gain(analytical_diagram, source=x, target=y) == (4.0 + 2.0 / 5.0)

def test_diagram_worst_case_peak_gain():
    x, y, z = sympy.symbols('x y z')
    diagram = series_diagram([
        parallel_diagram([
            DirectFormI.from_model(
                # F(z^-1) = 2 / (1 - 0.5 z^-1)
                # h[n] = 2 0.5^n u[n]
                # WCPG = 2 sum(|0.5^n|)
                # WCPG = 2 / (1 - 0.5) = 4
                signal.dlti([2, 0], [1, -0.5])
            ),
            DirectFormI.from_model(
                # F(z^-1) = 0.5 z^-1 / (1 - z^-1 + 0.5 z^-2)
                # F(z^-1; a = 1 / sqrt(2), b = pi/4) =
                #   a sin(b) z^-1 / (1 - 2 a cos(b) z^-1 + a^2 z^-2)
                # h[n] = a^n sin(b n) u[n]
                # WCPG = sum(|a^n sin(b n)|)
                # WCPG = sum(a^n |sin(b n)|)
                # WCPG = sum(a^(2k+1) |sin(b (2k+1))|) +
                #        sum(a^(4k+2) |sin(b (4k+2))|)
                # WCPG = a/sin(b) sum((a^2)^k) +
                #        a^2 sum((a^4)^k)
                # WCPG = 1 / (1 - a^2) + a^2 / (1 - a^4)
                # WCPG = 1 + 0.666666... = 5 / 3
                signal.dlti([0.5, 0], [1, -1, 0.5])
            ),
        ], output=y),
        DirectFormI.from_model(
            # F(z^-1) = 1 / (1 - 0.1 z^-1)
            # h[n] = 0.1^n u[n]
            # WCPG = sum(|0.1^n|)
            # WCPG = 1 / (1 - 0.1) = 1 / 0.9
            signal.dlti([1, 0], [1, -0.1])
        )
    ], input_=x, output=z)

    analytical_diagram = analytic_diagram(diagram)
    assert worst_case_peak_gain(analytical_diagram) == (4.0 + 5.0 / 3.0) * 10.0 / 9.0
    assert worst_case_peak_gain(analytical_diagram, source=y, target=z) == 10.0 / 9.0


def test_diagram_output_range():
    x, y, z = sympy.symbols('x y z')
    diagram = series_diagram([
        parallel_diagram([
            DirectFormI.from_model(
                # F(z^-1) = 2 / (1 - 0.5 z^-1)
                # DC GAIN = F(z = 1)
                # DC GAIN = 2 / (1 - 0.5) = 4
                # h[n] = 2 0.5^n u[n]
                # WCPG = 2 sum(|0.5^n|)
                # WCPG = 2 / (1 - 0.5) = 4
                signal.dlti([2, 0], [1, -0.5])
            ),
            DirectFormI.from_model(
                # F(z^-1) = 0.5 z^-1 / (1 - z^-1 + 0.5 z^-2)
                # F(z^-1; a = 1 / sqrt(2), b = pi/4) =
                #   a sin(b) z^-1 / (1 - 2 a cos(b) z^-1 + a^2 z^-2)
                # DC GAIN = F(z = 1) = 0.5 / (1 - 1 + 0.5) = 1
                # h[n] = a^n sin(b n) u[n]
                # WCPG = sum(|a^n sin(b n)|)
                # WCPG = sum(a^n |sin(b n)|)
                # WCPG = sum(a^(2k+1) |sin(b (2k+1))|) +
                #        sum(a^(4k+2) |sin(b (4k+2))|)
                # WCPG = a/sin(b) sum((a^2)^k) +
                #        a^2 sum((a^4)^k)
                # WCPG = 1 / (1 - a^2) + a^2 / (1 - a^4)
                # WCPG = 1 + 0.666666... = 5 / 3
                signal.dlti([0.5, 0], [1, -1, 0.5])
            )
        ], output=y),
        DirectFormI.from_model(
            # F(z^-1) = 1 / (1 - 0.1 z^-1)
            # DC GAIN = F(z = 1) =
            # DC GAIN = 1 / (1 - 0.1) = 1 / 0.9
            # h[n] = 0.1^n u[n]
            # WCPG = sum(|0.1^n|)
            # WCPG = 1 / (1 - 0.1) = 1 / 0.9
            signal.dlti([1, 0], [1, -0.1])
        )
    ], input_=x, output=z)

    input_mean = 0.5
    input_delta = 0.5
    input_range = interval(
        input_mean - input_delta,
        input_mean + input_delta
    )
    dcg = (4.0 + 1.0) / 0.9
    wcpg = (4.0 + 5.0 / 3.0) / 0.9
    expected_output_mean = dcg * input_mean
    expected_output_delta = wcpg * input_delta
    expected_output_range = interval(
        expected_output_mean - expected_output_delta,
        expected_output_mean + expected_output_delta,
    )
    analytical_diagram = analytic_diagram(diagram)
    assert output_range(analytical_diagram, input_range) == expected_output_range
    assert output_range(analytical_diagram, input_range, source=y, target=z) == input_range / 0.9


def test_diagram_error_bounds():
    x, y, z = sympy.symbols('x y z')

    diagram = series_diagram([
        parallel_diagram([
            DirectFormI.from_model(
                # F(z^-1) = z^-1 / (1 + 0.5 z^-1)
                signal.dlti([0.25], [1, 0.5])
            ),
            DirectFormI.from_model(
                # F(z) = z^-1 / (1 + z^-1 + 0.5 z^-2)
                signal.dlti([0.25, 0], [1, -1, 0.5])
            )
        ], output=y),
        DirectFormI.from_model(
            # F(z) = z^-1 / (1 - 0.1 z^-1)
            signal.dlti([1], [1, -0.1])
        )
    ], input_=x, output=z)

    with FixedFormatArithmeticLogicUnit(
        format_=Q(1, 7), allows_overflow=False,
        rounding_method=nearest_integer,
    ):
        analytical_diagram = analytic_diagram(diagram, {x: interval(-0.5, 0.5)})
        assert error_bounds(analytical_diagram, source=y) in error_bounds(analytical_diagram)


def test_diagram_signal_processing():
    x, y, z = sympy.symbols('x y z')
    F = DirectFormI.from_model(
        # F(z^-1) = z^-1 / (1 - 0.5 z^-1)
        signal.dlti([1], [1, -0.5])
    )
    G = DirectFormI.from_model(
        # F(z) = z^-1 / (1 + z^-1 + 0.5 z^-2)
        signal.dlti([1, 0], [1, 1, 0.5])
    )
    H = DirectFormI.from_model(
        # F(z) = z^-1 / (1 - 0.1 z^-1)
        signal.dlti([1], [1, -0.1])
    )
    diagram = series_diagram([
        parallel_diagram([F, G], output=y), H
    ], input_=x, output=z)
    model = series_composition([
        parallel_composition([F.model, G.model]), H.model
    ])
    _, (impulse_response,) = model.impulse(n=100)

    f = signal_processing_function(diagram)
    assert_allclose(f([1.] + [0.] * 99), impulse_response)

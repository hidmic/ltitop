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

from ltitop.models.composites import parallel_composition
from ltitop.models.composites import series_composition
from ltitop.models.composites import parallel_decomposition
from ltitop.models.composites import series_decomposition

import numpy as np
import scipy.signal as signal
from numpy.testing import assert_almost_equal

def test_series_composition():
    head = signal.lti([1], [1, 0.5])
    tail = signal.lti([1, 0], [1, 1, 0.5])
    series = series_composition([head, tail])
    expected_series = signal.lti([1, 0], [1, 1.5, 1, 0.25])
    w, Hs = series.freqresp()
    _, He = expected_series.freqresp(w=w)
    assert_almost_equal(Hs, He)

def test_series_decomposition():
    series = signal.lti([1, 0], [1, 1.5, 1, 0.25])
    head, tail = series_decomposition(series, 2)
    w, Hs = series.freqresp()
    _, He = head.freqresp(w=w)
    _, Ht = tail.freqresp(w=w)
    assert_almost_equal(Hs, He * Ht)

def test_parallel_composition():
    left = signal.lti([1], [1, 0.5])
    right = signal.lti([1, 0], [1, 1, 0.5])
    parallel = parallel_composition([left, right])
    expected_parallel = signal.lti([2, 1.5, 0.5], [1, 1.5, 1, 0.25])
    w, Hp = parallel.freqresp()
    _, He = expected_parallel.freqresp(w=w)
    assert_almost_equal(Hp, He)

def test_parallel_decomposition():
    parallel = signal.lti([1], [1, -0.5])
    left, right = parallel_decomposition(parallel, 2)
    w, Hp = parallel.freqresp()
    _, Hl = left.freqresp(w=w)
    _, Hr = right.freqresp(w=w)
    assert_almost_equal(Hp, Hl + Hr)

    parallel = signal.lti([2, 1.5, 0.5], [1, 1.5, 1, 0.25])
    left, right = parallel_decomposition(parallel, 2, tol=1e-15)
    w, Hp = parallel.freqresp()
    _, Hl = left.freqresp(w=w)
    _, Hr = right.freqresp(w=w)
    assert_almost_equal(Hp, Hl + Hr)

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
import scipy.signal as signal

def transform(func, model, **kwargs):
    model = model._as_zpk()
    return signal.ltisys.ZerosPolesGain(*func(
        model.zeros, model.poles, model.gain, **kwargs
    ))

lowpass_to_lowpass = functools.partial(transform, signal.lp2lp_zpk)
lowpass_to_highpass = functools.partial(transform, signal.lp2hp_zpk)
lowpass_to_bandpass = functools.partial(transform, signal.lp2bp_zpk)
lowpass_to_bandstop = functools.partial(transform, signal.lp2bs_zpk)

def discretize(model, dt):
    model = model._as_zpk()
    z, p, k = signal.bilinear_zpk(
        model.zeros, model.poles, model.gain, 1/dt
    )
    return signal.ltisys.ZerosPolesGainDiscrete(z, p, k, dt=dt)

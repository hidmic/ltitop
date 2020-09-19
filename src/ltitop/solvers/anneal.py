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

from simanneal import Annealer

import ltitop.solvers.gp as gp

def cast(*args, **kwargs):
    toolbox = gp.cast(*args, **kwargs)
    class ImplementationAnnealer(Annealer):
        def move(self):
            self.state = toolbox.mutate(self.state)

        def energy(self):
            print(self.state)
            self.state.fitness.values = \
                toolbox.evaluate(self.state)
            return sum(w * v for w, v in zip(
                self.state.fitness.weights,
                self.state.fitness.values))

        def anneal(self):
            state, _ = super().anneal()
            return toolbox.cast(state), state.fitness.values

    return ImplementationAnnealer(toolbox.individual())

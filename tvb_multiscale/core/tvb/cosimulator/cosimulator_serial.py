# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.

.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator


class CoSimulatorSerial(CoSimulator):

    simulate_spiking_simulator = None

    def configure(self, full_configure=True):
        super(CoSimulatorSerial, self).configure(full_configure=full_configure)
        if self.simulate_spiking_simulator is not None and self.synchronization_n_step == 0:
            self._cosimulation_flag = True
            self.synchronization_time = self.simulation_length
            self.synchronization_n_step = int(np.round(self.synchronization_time / self.integrator.dt))

    def _loop_update_cosim_history(self, step, state):
        if self.cosim_history is not None:
            super(CoSimulatorSerial, self)._loop_update_cosim_history(step, state)
        else:
            super(CoSimulatorSerial, self)._loop_update_history(step, np.copy(state))
        return state

    def _update_cosim_history(self, current_steps, cosim_updates):
        if self.cosim_history is not None:
            super(CoSimulatorSerial, self)._update_cosim_history(current_steps, cosim_updates)

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        steps_performed = \
            super(CoSimulatorSerial, self)._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation, **kwds)
        if cosimulation and self.simulate_spiking_simulator is not None:
            if self.n_tvb_steps_sent_to_cosimulator_at_last_synch is None:
                steps_to_run = steps_performed
            else:
                steps_to_run = self.n_tvb_steps_sent_to_cosimulator_at_last_synch
            self.log.info("Simulating the spiking network for %d time steps...", steps_to_run)
            self.simulate_spiking_simulator(steps_to_run * self.integrator.dt)
        return steps_performed

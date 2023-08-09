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

    §   §
"""

import time

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator


class CoSimulatorSerial(CoSimulator):

    simulate_spiking_simulator = None

    def run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        self.n_tvb_steps_ran_since_last_synch = \
            super(CoSimulatorSerial, self).run_for_synchronization_time(
                ts, xs, wall_time_start, self.get_cosim_updates(cosimulation), cosimulation=False, **kwds)
        if self.simulate_spiking_simulator is not None:
            self.log.info("Simulating the spiking network for %d time steps..." %
                          self.n_tvb_steps_ran_since_last_synch)
            self.simulate_spiking_simulator(
                np.around(self.n_tvb_steps_ran_since_last_synch * self.integrator.dt,
                          decimals=self._number_of_dt_decimals).item())
        return self.send_cosim_coupling(cosimulation), self.n_tvb_steps_ran_since_last_synch

    def run_cosimulation(self, ts, xs, wall_time_start, advance_simulation_for_delayed_monitors_output=True, **kwds):
        """Convenience method to run cosimulation for serial cosimulation."""
        simulation_length = self.simulation_length
        synchronization_time = self.synchronization_time
        if advance_simulation_for_delayed_monitors_output:
            simulation_length += synchronization_time
        synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
        if not self.n_tvb_steps_ran_since_last_synch:
            self.n_tvb_steps_ran_since_last_synch = synchronization_n_step
        simulated_steps = 0
        remaining_steps = int(numpy.round(simulation_length / self.integrator.dt))
        # Send TVB's initial condition to spikeNet!:
        self.send_cosim_coupling(True)
        self._tic = time.time()
        while remaining_steps > 0:
            self.synchronization_n_step = numpy.minimum(remaining_steps, synchronization_n_step)
            self.n_tvb_steps_ran_since_last_synch = \
                self.run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=True, **kwds)[-1]
            simulated_steps += self.n_tvb_steps_ran_since_last_synch
            remaining_steps -= self.n_tvb_steps_ran_since_last_synch
            self._log_print_progress_message(simulated_steps, simulation_length)
        self.synchronization_n_step = int(synchronization_n_step)  # restore the configured value
        self.simulation_length = simulation_length                 # restore the actually implemented value

    def run(self, **kwds):
        """Convenience method to call the CoSimulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        self.simulation_length = kwds.pop("simulation_length", self.simulation_length)
        asfdmo = kwds.pop("advance_simulation_for_delayed_monitors_output", True)
        if self._cosimulation_flag:
            self.run_cosimulation(ts, xs, wall_time_start,
                                  advance_simulation_for_delayed_monitors_output=asfdmo,
                                  **kwds)
        else:
            self.run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=False, **kwds)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))

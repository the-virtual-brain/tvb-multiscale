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

import time

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator


class CoSimulatorSerial(CoSimulator):

    simulate_spiking_simulator = None

    def _get_cosim_updates(self, cosimulation=True):
        cosim_updates = None
        if cosimulation and self.input_interfaces:
            # Get the update data from the other cosimulator
            cosim_updates = self.input_interfaces(self.good_cosim_update_values_shape)
            isnans = np.isnan(cosim_updates[-1])
            if np.all(isnans):
                cosim_updates = None
                self.log.warning("No or all NaN valued cosimulator updates at time step %d!" % self.current_step)
            elif np.any(isnans):
                msg = "NaN values detected in cosimulator updates at time step %d!" % self.current_step
                self.log.error(msg)
                raise Exception(msg)
        return cosim_updates

    def _send_cosim_coupling(self, cosimulation=True):
        outputs = []
        if cosimulation and self.output_interfaces and self.n_tvb_steps_ran_since_last_synch > 0:
            if self.output_interfaces.number_of_interfaces:
                # Send the data to the other cosimulator
                outputs = \
                    self.output_interfaces(self.loop_cosim_monitor_output(self.n_tvb_steps_ran_since_last_synch))
            self.n_tvb_steps_sent_to_cosimulator_at_last_synch = int(self.n_tvb_steps_ran_since_last_synch)
            self.n_tvb_steps_ran_since_last_synch = 0
        return outputs

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        steps_performed = \
            super(CoSimulatorSerial, self)._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation, **kwds)
        if self.simulate_spiking_simulator is not None:
            steps_to_run = np.where(self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                    self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                    steps_performed).item()
            self.log.info("Simulating the spiking network for %d time steps..." % steps_to_run)
            self.simulate_spiking_simulator(np.around(steps_to_run * self.integrator.dt,
                                                      decimals=self._number_of_dt_decimals).item())
        return steps_performed

    def _run_cosimulation(self, ts, xs, wall_time_start, advance_simulation_for_delayed_monitors_output=True, **kwds):
        simulated_steps = 0
        simulation_length = self.simulation_length
        synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
        if self.n_tvb_steps_ran_since_last_synch is None:
            self.n_tvb_steps_ran_since_last_synch = synchronization_n_step
        remaining_steps = int(np.round(simulation_length / self.integrator.dt))
        self._tic = time.time()
        while remaining_steps > 0:
            self.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
            steps_performed = \
                self._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=True, **kwds)
            simulated_steps += steps_performed
            remaining_steps -= steps_performed
            self.n_tvb_steps_ran_since_last_synch += steps_performed
            self._log_print_progress_message(simulated_steps, simulation_length)
        self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
        if self._cosimulation_flag and advance_simulation_for_delayed_monitors_output:
            # Run once more for synchronization steps in order to get the full delayed monitors' outputs:
            remaining_steps = \
                int(np.round((simulation_length + self.synchronization_time - simulated_steps*self.integrator.dt)
                             / self.integrator.dt))
            if remaining_steps:
                self.log.info("Simulating for synchronization excess time %0.3f...",
                              remaining_steps * self.integrator.dt)
                synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
                self.synchronization_n_step = np.minimum(synchronization_n_step, remaining_steps)
                self._run_for_synchronization_time(ts, xs, wall_time_start,
                                                   cosimulation=False, **kwds)  # Run only TVB
                self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
        self.simulation_length = simulation_length  # recover the configured value

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
            self._run_cosimulation(ts, xs, wall_time_start,
                                   advance_simulation_for_delayed_monitors_output=asfdmo,
                                   **kwds)
        else:
            self._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=False, **kwds)
        for i in range(len(ts)):
            ts[i] = np.array(ts[i])
            xs[i] = np.array(xs[i])
        return list(zip(ts, xs))

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

import ray

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel


class CoSimulatorSerialRay(CoSimulatorSerial):

    spiking_simulator = None

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        if self.spiking_simulator is not None:
            self.spiking_simulator.block_run
        steps_performed = \
            super(CoSimulatorSerial, self)._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation, **kwds)
        if self.spiking_simulator is not None:
            steps_to_run = np.where(self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                    self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                    steps_performed).item()
            self.log.info("Simulating the spiking network for %d time steps..." % steps_to_run)
            self.spiking_simulator.Run(np.around(steps_to_run * self.integrator.dt,
                                                 decimals=self._number_of_dt_decimals).item())
        return steps_performed


class CoSimulatorParallelRay(CoSimulatorParallel):

    spiking_simulator = None

    def _send_cosim_coupling(self, cosimulation=True, outputs=[], block=False):
        if len(outputs) == 0:
            return super(CoSimulatorParallelRay, self)._send_cosim_coupling(cosimulation)
        else:
            return self.output_interfaces(block=block)

    def _get_cosim_updates(self, cosimulation=True, block=False, cosim_updates=None):
        # Get the update data from the other cosimulator, including any transformations
        if cosimulation and self.input_interfaces:
            if cosim_updates is None:
                cosim_updates = self.input_interfaces(self.good_cosim_update_values_shape, block=block)
            else:
                for cosim_update in cosim_updates:
                    if isinstance(cosim_update, ray._raylet.ObjectRef):
                        cosim_updates = self.input_interfaces(self.good_cosim_update_values_shape, block=block)
                        break
        if cosim_updates is not None and np.all(np.isnan(cosim_updates[-1])):
            cosim_updates = None
        return cosim_updates

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        # Loop of integration for synchronization_time
        # spikeNet is the bottleneck without MPI communication.
        # The order of events for spikeNet is:
        # 1. spikeNet output, 2. spikeNet input, 3. spikeNet integration
        # So, we submit these remote jobs in that order:

        if cosimulation and self.spiking_simulator is not None:
            # 1. Start processing TVB data and send them to spikeNet when the latter is ready
            # Transform and send TVB -> spikeNet
            tvb_to_spikeNet_locks = self._send_cosim_coupling(self._cosimulation_flag)  # NON BLOCKING

            # -----------------BLOCK at this point for the spikeNet simulator to finish integrating----------------:
            if self.spiking_simulator.is_running:
                self.spiking_simulator.block_run

            # 2. Get data from spikeNet and start processing them
            # Receive and transform TVB <- spikeNet
            cosim_updates = self._get_cosim_updates(cosimulation, block=False)  # NON BLOCKING

            # 3. Start simulating spikeNet as long as the TVB data have arrived.
            # Integrate spikeNet
            self.log.info("Simulating the spiking network for %d time steps...",
                          self.n_tvb_steps_sent_to_cosimulator_at_last_synch)
            self.spiking_simulator.RunLock(
                    self.n_tvb_steps_sent_to_cosimulator_at_last_synch * self.integrator.dt,
                    self._send_cosim_coupling(self._cosimulation_flag, tvb_to_spikeNet_locks, block=True)
                )  # tvb_to_spikeNet_locks are used in order to block spikeNet simulator from starting to integrate
        else:
            cosim_updates = None

        # 4. Start simulating TVB as long as the spikeNet data have been processed.
        # Integrate TVB
        current_step = int(self.current_step)
        for data in self(cosim_updates=self._get_cosim_updates(cosimulation,
                                                               block=True, cosim_updates=cosim_updates),  # BLOCKING
                         **kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        steps_performed = self.current_step - current_step
        return steps_performed

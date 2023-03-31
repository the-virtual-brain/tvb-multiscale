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

from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator

import numpy as np
import time, sys
from threading import Thread

class CoSimulatorParallel(CoSimulator):

    pass


class CoSimulatorMPI(CoSimulatorParallel):

    pass
    # def _run_cosimulation(self, ts, xs, wall_time_start, advance_simulation_for_delayed_monitors_output=True, **kwds):
    #     super(CoSimulatorMPI, self)._run_cosimulation(ts, xs, wall_time_start,
    #                                                   advance_simulation_for_delayed_monitors_output, **kwds)
    #     self.logger.info(" TVB finish")
    #     if self.n_output_interfaces:
    #         logger.info('end comm send')
    #         self.output_interfaces[0].end_mpi()
    #     if self.n_input_interfaces:
    #         logger.info('end comm receive')
    #         self.input_interfaces[0].end_mpi()
    #     self.MPI.Finalize()  # ending with MPI
    #     self.logger.info("TVB exit")

class CoSimulatorNetpyne(CoSimulator):

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):

        self.synchronize_spiking_simulator()

        # root node performs co-simulation updates in both directions (spiking <-> TVB) and calculates spiking simulation length for this iteration
        if self.isRootNode():
            self._send_cosim_coupling(self._cosimulation_flag)
            cosim_updates = self._get_cosim_updates(cosimulation)

            n_steps = self._prepare_simulation(cosim_updates=cosim_updates)
            steps_to_run_spiking = np.where(self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                            self.n_tvb_steps_sent_to_cosimulator_at_last_synch,
                                            n_steps).item()
            spiking_simulation_length = np.around(steps_to_run_spiking * self.integrator.dt,
                                                  decimals=self._number_of_dt_decimals).item()
        else:
            spiking_simulation_length, n_steps = (0, 0) # just a placeholder

        # transfer to other MPI nodes
        spiking_simulation_length, n_steps = self.synchronize_spiking_simulator(
            gatherSimData=False,
            additionalData = [spiking_simulation_length, n_steps]
        )
        n_steps = int(n_steps)

        if self.isRootNode():
            # root node performs TVB simulation and spiking simulation for its share of spiking cells in separate threads. Spiking cells load balancing between MPI nodes is done the way that root node gets minority of it, so it should have enough idle time to bypass the curse of global interpreter lock, thus justifying the threading approach

            def runTVB():
                data = self(cosim_updates=cosim_updates, n_steps=n_steps, recompute_steps=False, **kwds)
                self._update_monitors_with_new_output(ts, xs, data)

            tvbThread = Thread(target=runTVB)
            tvbThread.start()
            self.simulate_spiking_simulator(spiking_simulation_length)
            tvbThread.join()
        else:
            # the rest of nodes run only spiking simulation
            self.simulate_spiking_simulator(spiking_simulation_length)

        sys.stdout.flush() # workaround for logs congestion issue
        return n_steps

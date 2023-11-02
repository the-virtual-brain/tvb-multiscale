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

from tvb.basic.neotraits.api import Int

from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel


class CoSimulatorParallelRay(CoSimulatorParallel):

    min_idelay_sync_n_step_ratio = Int(
        label="min_idelay_synch_n_step_ratio",
        choices=(2, ),
        default=2,
        required=True,
        doc="""min_idelay to synchronization_n_step ratio, 
               i.e., an integer value defining how many times smaller should the synchronization time be 
               compared to the minimum delay time in integration time steps.
               For the moment we limit it to 1 (synchronization_time = min_delay) 
               or 2 (synchronization_time = min_delay/2)""")

    _run_serially = False

    def __init__(self, **kwargs):
        super(CoSimulatorParallelRay, self).__init__(**kwargs)
        self.min_idelay_sync_n_step_ratio = 2
        self._run_serially = False

    def run_for_synchronization_time(self, ts, xs, wall_time_start,
                                     spiking_network,
                                     tvb_to_spikeNet_trans_interfaces, spikeNet_to_tvb_trans_interfaces,
                                     input_tvb_to_trans_cosim_updates=None,
                                     input_ref_trans_to_spikeNet_cosim_updates=list(),
                                     input_spikeNet_to_trans_cosim_updates=None,
                                     input_ref_trans_to_tvb_cosim_updates=list()):
        """Function for cosimulating for one loop of synchronization time.
           It could be the highest level possible ENTRYPOINT for a Ray parallel cosimulation.
           The ENTRYPOINT here is just the cosimulation updates' data,
           which are "thrown over the wall" for the necessary data exchanges.
           All processes can work in parallel,
           since the transformers work on time intervals other than those simulated by the cosimulators.
        """
        # NON-BLOCKING:
        # spikeNet t -> t + Tsync
        # Simulate spikeNet for times [t, t + Tsync] = [t, t + min_tvb_delay/2]
        # with or without TVB coupling inputs of times [t, t + min_tvb_delay/2]
        # or TVB state inputs of times [t - min_tvb_delay, t - min_tvb_delay/2]
        input_trans_to_spikeNet_cosim_updates = None
        if self._run_serially:
            input_trans_to_spikeNet_cosim_updates = input_ref_trans_to_spikeNet_cosim_updates
        elif len(input_ref_trans_to_spikeNet_cosim_updates):
            input_trans_to_spikeNet_cosim_updates = \
                tvb_to_spikeNet_trans_interfaces.rayget(input_ref_trans_to_spikeNet_cosim_updates)
        spiking_network.input_interfaces(input_trans_to_spikeNet_cosim_updates)
        spikeNet_run_ref = spiking_network.Run(self.synchronization_time)
        if self._run_serially:
            ray.get(spikeNet_run_ref)

        # NON-BLOCKING:
        # Transform spikeNet -> TVB updates of times [t - Tsync, t] = [t - min_tvb_delay/2, t]...
        if input_spikeNet_to_trans_cosim_updates is not None:
            # ...if any:
            ref_trans_to_tvb_cosim_updates = spikeNet_to_tvb_trans_interfaces(input_spikeNet_to_trans_cosim_updates)
            if self._run_serially:
                ref_trans_to_tvb_cosim_updates = \
                    spikeNet_to_tvb_trans_interfaces.rayget(ref_trans_to_tvb_cosim_updates)
        else:
            ref_trans_to_tvb_cosim_updates = []

        # NON-BLOCKING:
        # Transform TVB -> spikeNet couplings of times [t + Tsync, t + 2*Tsync]
        # = [t + min_tvb_delay/2, t + min_tvb_delay]...
        # ...or TVB -> spikeNet state of times [t - Tsync, t] = [t - min_tvb_delay/2, t]
        if input_tvb_to_trans_cosim_updates is not None:
            # ...if any:
            ref_trans_to_spikeNet_cosim_updates = tvb_to_spikeNet_trans_interfaces(input_tvb_to_trans_cosim_updates)
            if self._run_serially:
                ref_trans_to_spikeNet_cosim_updates = \
                    spikeNet_to_tvb_trans_interfaces.rayget(ref_trans_to_spikeNet_cosim_updates)
        else:
            ref_trans_to_spikeNet_cosim_updates = []

        # BLOCKING:
        # Simulate TVB for times [t, t + Tsync] = [t, t + min_tvb_delay/2]
        # with or without spikeNet update inputs of times [t - 2*Tsync, t - Tsync]
        # = [t - min_tvb_delay/, t - min_tvb_delay/2]
        input_trans_to_tvb_cosim_updates = None
        if self._run_serially:
            input_trans_to_tvb_cosim_updates = input_ref_trans_to_tvb_cosim_updates
        elif len(input_ref_trans_to_tvb_cosim_updates):
            input_trans_to_tvb_cosim_updates = \
                spikeNet_to_tvb_trans_interfaces.rayget(input_ref_trans_to_tvb_cosim_updates)
        tvb_to_trans_cosim_updates = super(CoSimulatorParallelRay, self).run_for_synchronization_time(
            ts, xs, wall_time_start, input_trans_to_tvb_cosim_updates, cosimulation=True)

        # BLOCK for spikeNet to finish running:
        if not self._run_serially:
            ray.get(spikeNet_run_ref)
        spikeNet_to_trans_cosim_updates = spiking_network.output_interfaces()

        # Now time has become t + Tsync!
        return tvb_to_trans_cosim_updates, ref_trans_to_spikeNet_cosim_updates, \
               spikeNet_to_trans_cosim_updates, ref_trans_to_tvb_cosim_updates, \
               input_trans_to_spikeNet_cosim_updates, input_trans_to_tvb_cosim_updates  # Only for debugging

    def run_cosimulation(self, ts, xs, wall_time_start,
                         spiking_network, tvb_to_spikeNet_trans_interfaces, spikeNet_to_tvb_trans_interfaces,
                         advance_simulation_for_delayed_monitors_output=True,
                         **kwds):
        """Convenience method to run cosimulation for parallel Ray cosimulation."""

        self._run_serially = kwds.pop("run_serially", False)

        # Get and store the configured values of simulator's time parameters:
        simulation_length = self.simulation_length
        synchronization_time = self.synchronization_time
        synchronization_n_step = int(self.synchronization_n_step)

        # Given that we that data require two synchronization periods
        # to be communicated from one cosimulator to the other,
        # we set the relative_output_time_steps accordingly,
        # i.e., to sample the output from two times the synchronization times in the past...:
        self.relative_output_time_steps = self.synchronization_n_step
        # ...and we advance the simulation time by two times the synchronization times,
        if advance_simulation_for_delayed_monitors_output:
            simulation_length += (self.synchronization_n_step + self.relative_output_time_steps) * self.integrator.dt

        # Send TVB's initial condition to spikeNet!:
        self.relative_output_interfaces_time_steps = 0
        # TVB initial condition cosimulation data towards spikeNet
        # - of TVB coupling for [0, Tsync] if INTERFACE_COUPLING_MODE == "TVB"
        # - of TVB state for [-Tsync, 0] if INTERFACE_COUPLING_MODE == "spikeNet"
        tvb_to_trans_cosim_updates = self.send_cosim_coupling(True)
        # spikeNet initial condition update towards TVB:
        spikeNet_to_trans_cosim_updates = None  # data
        # Transformer to TVB initial condition:
        ref_trans_to_tvb_cosim_updates = []  # ray reference objects
        # Transformer to spikeNet initial condition:
        ref_trans_to_spikeNet_cosim_updates = []  # ray reference objects
        # TODO: Deal with heterogeneous interfaces in terms of coupling!
        if self.output_interfaces.interfaces[0].coupling_mode == "TVB":
            # Advancement in time for the simulation to follow:
            self.relative_output_interfaces_time_steps = synchronization_n_step
            # If this is a TVB coupling interface,
            # ...transform the initial condition for [0, Tsync] = [0 min_tvb_delay/2] from TVB to spikeNet:
            if tvb_to_trans_cosim_updates is not None:
                # ...if any, get the ray ref objects:
                ref_trans_to_spikeNet_cosim_updates = tvb_to_spikeNet_trans_interfaces(tvb_to_trans_cosim_updates)
            # ...and advance the data sent from TVB towards the transformer,
            # by Tsync = min_tvb_delay / 2
            # TVB initial condition cosimulation coupling towards spikeNet
            # for [Tsync, 2*Tsync] = [min_tvb_delay/2, min_tvb_delay]
            tvb_to_trans_cosim_updates = self.send_cosim_coupling(True)  # data
        else:
            # If this is a spikeNet coupling interface,
            # ...get the past state from TVB towards the transformer,
            # by Tsync = -min_tvb_delay / 2
            # TVB initial condition cosimulation state towards spikeNet
            # for [-2*Tsync, Tsync] = [-min_tvb_delay, -min_tvb_delay/2]
            self.relative_output_interfaces_time_steps = -synchronization_n_step
            tvb_to_trans_cosim_updates_past = self.send_cosim_coupling(True)  # data
            # ...and transform it:
            if tvb_to_trans_cosim_updates_past is not None:
                # ...if any, get the ray ref objects:
                ref_trans_to_spikeNet_cosim_updates = tvb_to_spikeNet_trans_interfaces(tvb_to_trans_cosim_updates_past)
            # Cancel the time delay for the simulation to follow:
            self.relative_output_interfaces_time_steps = 0
        if self._run_serially:
            ref_trans_to_spikeNet_cosim_updates = \
                tvb_to_spikeNet_trans_interfaces.rayget(ref_trans_to_spikeNet_cosim_updates)

        # Prepare loop...
        simulated_steps = 0
        remaining_steps = int(np.round(simulation_length / self.integrator.dt))
        if not self.n_tvb_steps_ran_since_last_synch:
            self.n_tvb_steps_ran_since_last_synch = synchronization_n_step
        # ...and loop:
        self._tic = time.time()
        while remaining_steps > 0:
            # Set the remaining steps as simulation time,
            # if it is less than the original synchronization time:
            self.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
            time_to_simulate = self.integrator.dt * self.synchronization_n_step
            self.synchronization_time = time_to_simulate
            tvb_to_trans_cosim_updates, ref_trans_to_spikeNet_cosim_updates, \
            spikeNet_to_trans_cosim_updates, ref_trans_to_tvb_cosim_updates = \
                self.run_for_synchronization_time(
                    ts, xs, wall_time_start,
                    spiking_network, tvb_to_spikeNet_trans_interfaces, spikeNet_to_tvb_trans_interfaces,
                    tvb_to_trans_cosim_updates, ref_trans_to_spikeNet_cosim_updates,
                    spikeNet_to_trans_cosim_updates, ref_trans_to_tvb_cosim_updates)[:4]
            simulated_steps += self.n_tvb_steps_ran_since_last_synch
            remaining_steps -= self.n_tvb_steps_ran_since_last_synch
            self._log_print_progress_message(simulated_steps, simulation_length)

        # Restore simulator time parameters:
        self.simulation_length = simulated_steps * self.integrator.dt  # restore the actually implemented value
        self.synchronization_n_step = int(synchronization_n_step)  # restore the configured value
        self.synchronization_time = synchronization_time


class CoSimulatorParallelSpikeNetRay(CoSimulatorParallel):

    min_idelay_sync_n_step_ratio = Int(
        label="min_idelay_synch_n_step_ratio",
        choices=(1, ),
        default=1,
        required=True,
        doc="""min_idelay to synchronization_n_step ratio, 
               i.e., an integer value defining how many times smaller should the synchronization time be 
               compared to the minimum delay time in integration time steps.
               For the moment we limit it to 1 (synchronization_time = min_delay) 
               or 2 (synchronization_time = min_delay/2)""")

    _run_serially = False

    def __init__(self, **kwargs):
        super(CoSimulatorParallelSpikeNetRay, self).__init__(**kwargs)
        self.min_idelay_sync_n_step_ratio = 1
        self._run_serially = False

    def run_for_synchronization_time(self, ts, xs, wall_time_start,
                                     spiking_network,
                                     input_TVBtrans_to_spikeNet_cosim_updates=None,
                                     input_spikeNet_to_transTVB_cosim_updates=None):
        """Function for cosimulating for one loop of synchronization time.
           It could be the highest level possible ENTRYPOINT for a Ray parallel cosimulation.
           The ENTRYPOINT here is just the cosimulation updates' data,
           which are "thrown over the wall" for the necessary data exchanges.
           Spiking network simulation works in parallel with TVB simulation and transformer's operations.
        """
        # NON-BLOCKING:
        # spikeNet t -> t + Tsync
        # Simulate spikeNet for times [t, t + Tsync] = [t, t + min_tvb_delay]
        # with or without TVB coupling inputs of times [t, t + min_tvb_delay]
        # or TVB state inputs of times [t - min_tvb_delay, t]
        if input_TVBtrans_to_spikeNet_cosim_updates is not None:
            spiking_network.input_interfaces(input_TVBtrans_to_spikeNet_cosim_updates)
        spikeNet_run_ref = spiking_network.Run(self.synchronization_time)
        if self._run_serially:
            ray.get(spikeNet_run_ref)

        # BLOCKING:
        # 1. Transform spikeNet -> TVB updates of times [t - Tsync, t] = [t - min_tvb_delay, t], if any...
        # 2. TVB t -> t + Tsync
        #    Simulate TVB for times [t, t + Tsync] = [t, t + min_tvb_delay]
        #    with or without spikeNet update inputs of times [t - Tsync, t] = [t - min_tvb_delay, t]
        # 3. Transform TVB -> spikeNet couplings of times
        #    [t + Tsync, t + 2*Tsync] = [t + min_tvb_delay, t + 2*min_tvb_delay]...
        #    ...or TVB -> spikeNet state of times [t, t + Tsync]=[t, t + min_tvb_delay]
        TVBtrans_to_spikeNet_cosim_updates = super(CoSimulatorParallelSpikeNetRay, self).run_for_synchronization_time(
            ts, xs, wall_time_start, input_spikeNet_to_transTVB_cosim_updates, cosimulation=True)

        # BLOCK for spikeNet to finish running:
        if not self._run_serially:
            ray.get(spikeNet_run_ref)
        spikeNet_to_transTVB_cosim_updates = spiking_network.output_interfaces()

        # Now time has become t + Tsync!
        return TVBtrans_to_spikeNet_cosim_updates, spikeNet_to_transTVB_cosim_updates

    def run_cosimulation(self, ts, xs, wall_time_start, spiking_network,
                         advance_simulation_for_delayed_monitors_output=True, **kwds):

        """Convenience method to run cosimulation for parallel Ray cosimulation."""

        self._run_serially = kwds.pop("run_serially", False)

        # Get and store the configured values of simulator's time parameters:
        simulation_length = self.simulation_length
        synchronization_time = self.synchronization_time
        synchronization_n_step = int(self.synchronization_n_step)

        # Advance the simulation time by the synchronization times,
        if advance_simulation_for_delayed_monitors_output:
            simulation_length += self.synchronization_n_step * self.integrator.dt

        # Send TVB's initial condition to spikeNet via the TVB -> spikeNet transformer!:
        # TVB initial condition cosimulation data towards spikeNet
        # - of TVB coupling for [0, Tsync] if INTERFACE_COUPLING_MODE == "TVB"
        # - of TVB state for [-Tsync, 0] if INTERFACE_COUPLING_MODE == "spikeNet"
        TVBtrans_to_spikeNet_cosim_updates = self.send_cosim_coupling(True)
        # spikeNet initial condition update towards TVB via the transformer:
        spikeNet_to_transTVB_cosim_updates = None

        # Prepare loop...
        simulated_steps = 0
        remaining_steps = int(np.round(simulation_length / self.integrator.dt))
        if not self.n_tvb_steps_ran_since_last_synch:
            self.n_tvb_steps_ran_since_last_synch = synchronization_n_step
        # ...and loop:
        self._tic = time.time()
        while remaining_steps > 0:
            # Set the remaining steps as simulation time,
            # if it is less than the original synchronization time:
            self.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
            time_to_simulate = self.integrator.dt * self.synchronization_n_step
            self.synchronization_time = time_to_simulate
            TVBtrans_to_spikeNet_cosim_updates, spikeNet_to_transTVB_cosim_updates = \
                self.run_for_synchronization_time(
                    ts, xs, wall_time_start, spiking_network,
                    TVBtrans_to_spikeNet_cosim_updates, spikeNet_to_transTVB_cosim_updates)
            simulated_steps += self.n_tvb_steps_ran_since_last_synch
            remaining_steps -= self.n_tvb_steps_ran_since_last_synch
            self._log_print_progress_message(simulated_steps, simulation_length)

        # Restore simulator time parameters:
        self.simulation_length = simulated_steps * self.integrator.dt  # restore the actually implemented value
        self.synchronization_n_step = int(synchronization_n_step)  # restore the configured value
        self.synchronization_time = synchronization_time

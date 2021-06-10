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

from tvb.basic.neotraits.api import Attr
from tvb.contrib.cosimulation.cosimulator import CoSimulator as CoSimulatorBase
from tvb.contrib.cosimulation.cosim_monitors import RawCosim, CosimMonitorFromCoupling

from tvb_multiscale.core.config import LINE
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBOutputInterfaces
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBInputInterfaces


class CoSimulator(CoSimulatorBase):

    output_interfaces = Attr(
        field_type=TVBOutputInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    input_interfaces = Attr(
        field_type=TVBInputInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    PRINT_PROGRESSION_MESSAGE = True

    def _configure_synchronization_time(self):
        """This method will default synchronization time
           to be equal to the minimum delay time of connectivity,
           in case the user hasn't set it up until this point."""
        if self.synchronization_time == 0.0:
            self.synchronization_n_step = np.min(self.connectivity.idelays[np.nonzero(self.connectivity.weights)])
            self.synchronization_time = self.synchronization_n_step * self.integrator.dt
        super(CoSimulator, self)._configure_synchronization_time()

    def _configure_interfaces_vois_proxy_inds(self):
        """This method will
            - set the voi and spiking_proxy_inds of the CoSimulator, based on the predefined input and output interfaces,
            - configure all interfaces.
        """
        voi = []
        proxy_inds = []
        if self.output_interfaces:
            self.output_interfaces.dt = self.integrator.dt
            # Configure all TVB to Cosim interfaces:
            self.output_interfaces.configure()
            if self.n_output_interfaces:
                voi += self.output_interfaces.voi_unique.tolist()
                proxy_inds += self.output_interfaces.proxy_inds_unique.tolist()
        if self.input_interfaces:
            # Configure all Cosim to TVB interfaces:
            self.input_interfaces.configure()
            if self.n_input_interfaces:
                voi += self.input_interfaces.voi_unique.tolist()
                proxy_inds += self.input_interfaces.proxy_inds_unique.tolist()
        self.voi = np.unique(voi).astype(np.int)
        self.proxy_inds = np.unique(proxy_inds).astype(np.int)

    def _assert_cosim_monitors_vois_period(self):
        """This method will assert that
            - the period of all CosimMonitor instances set is equal to the integrator's dt.
            - there is at least one CosimMonitor instance set for any voi the output interfaces need,
         """
        periods = [cosim_monitor.period for cosim_monitor in self.cosim_monitors]
        assert np.allclose(periods, self.integrator.dt, 1e-6)
        if self.n_output_interfaces:
            n_cosim_monitors = len(self.cosim_monitors)
            assert n_cosim_monitors > 0
            for interface in self.output_interfaces.interfaces:
                assert n_cosim_monitors > interface.monitor_ind
                monitor = self.cosim_monitors[interface.monitor_ind]
                if interface.coupling_mode.upper() == "TVB":
                    assert isinstance(monitor, CosimMonitorFromCoupling)
                    cvar = self.model.cvar.tolist()
                    assert [cvar.index(voi) in monitor.voi for voi in interface.voi]
                else:
                    assert not isinstance(monitor, CosimMonitorFromCoupling)
                    assert [voi in monitor.voi for voi in interface.voi]

    def _configure_local_vois_and_proxy_inds_per_interface(self):
        """This method will set the local -per cosimulation and interface- voi and spiking_proxy_inds indices,
           based on the voi of each linked cosimulation monitor, for TVB to Cosimulator interfaces,
           and on the expected shape of ths cosimulation updates data for Cosimulator to TVB interfaces.
        """
        if self.n_output_interfaces:
            # Set the correct voi indices with reference to the linked TVB CosimMonitor, for each cosimulation:
            self.output_interfaces.set_local_indices(self.cosim_monitors)
        if self.n_input_interfaces:
            # Method to get the correct indices of voi and spiking_proxy_inds, for each cosimulation,
            # adjusted to the contents, shape etc of the cosim_updates,
            # based on TVB CoSimulators' vois and spiking_proxy_inds, i.e., good_cosim_update_values_shape
            self.input_interfaces.set_local_indices(self.voi, self.proxy_inds)

    def _configure_cosimulation(self):
        """This method will
           - set the synchronization time and number of steps,
           - check the time and the variable of interest are correct
           - create and initialize CosimHistory,
           - configure the cosimulation monitor
           - zero connectivity weights to/from nodes modelled exclusively by the other cosimulator
        """
        self.n_tvb_steps_ran_since_last_synch = None
        self.n_tvb_steps_sent_to_cosimulator_at_last_synch = None
        self.n_output_interfaces = self.output_interfaces.number_of_interfaces if self.output_interfaces else 0
        self.n_input_interfaces = self.input_interfaces.number_of_interfaces if self.input_interfaces else 0
        self._configure_interfaces_vois_proxy_inds()
        super(CoSimulator, self)._configure_cosimulation()
        if self._cosimulation_flag:
            self._assert_cosim_monitors_vois_period()
            self._configure_local_vois_and_proxy_inds_per_interface()

    def _prepare_stimulus(self):
        if self.simulation_length != self.synchronization_time:
            simulation_length = float(self.simulation_length)
            self.simulation_length = float(self.synchronization_time)
            super(CoSimulator, self)._prepare_stimulus()
            self.simulation_length = simulation_length
        else:
            super(CoSimulator, self)._prepare_stimulus()

    def _get_cosim_updates(self, cosimulation=True):
        cosim_updates = None
        if cosimulation and self.input_interfaces:
            # Get the update data from the other cosimulator
            cosim_updates = self.input_interfaces(self.good_cosim_update_values_shape)
            if np.all(np.isnan(cosim_updates[1])):
                cosim_updates = None
        return cosim_updates

    def _send_cosim_coupling(self, cosimulation=True):
        if cosimulation and self.output_interfaces and self.n_tvb_steps_ran_since_last_synch > 0:
            if self.output_interfaces.number_of_interfaces:
                # Send the data to the other cosimulator
                self.output_interfaces(
                    self.loop_cosim_monitor_output(self.n_tvb_steps_ran_since_last_synch))
            self.n_tvb_steps_sent_to_cosimulator_at_last_synch = int(self.n_tvb_steps_ran_since_last_synch)
            self.n_tvb_steps_ran_since_last_synch = 0

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        # Loop of integration for synchronization_time
        self._send_cosim_coupling(self._cosimulation_flag)
        current_step = int(self.current_step)
        for data in self(cosim_updates=self._get_cosim_updates(cosimulation), **kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        steps_performed = self.current_step - current_step
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        return steps_performed

    def _run_cosimulation(self, ts, xs, wall_time_start, **kwds):
        simulated_steps = 0
        simulation_length = self.simulation_length
        synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
        if self.n_tvb_steps_ran_since_last_synch is None:
            self.n_tvb_steps_ran_since_last_synch = synchronization_n_step
        remaining_steps = int(np.round(simulation_length / self.integrator.dt))
        while remaining_steps > 0:
            self.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
            steps_performed = \
                self._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=True, **kwds)
            simulated_steps += steps_performed
            remaining_steps -= steps_performed
            self.n_tvb_steps_ran_since_last_synch += steps_performed
            self.log.info("...%.3f%% completed!", 100 * (simulated_steps * self.integrator.dt) / simulation_length)
        self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
        if self._cosimulation_flag:
            # Run once more for synchronization steps in order to get the full delayed monitors' outputs:
            remaining_steps = \
                int(np.round((simulation_length + self.synchronization_time - simulated_steps*self.integrator.dt)
                             / self.integrator.dt))
            if remaining_steps:
                self.log.info("Simulating for synchronization excess time %0.3f...",
                              remaining_steps * self.integrator.dt)
                current_step = int(self.current_step)
                current_state = np.copy(self.current_state)
                synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
                self.synchronization_n_step = np.minimum(synchronization_n_step, remaining_steps)
                self._run_for_synchronization_time(ts, xs, wall_time_start,
                                                   cosimulation=False, **kwds)  # Run only TVB
                self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
                # Revert the current_step and current_state to those before the excess synchronization time
                self.current_step = int(current_step)
                self.current_state = np.copy(current_state)
        self.simulation_length = simulation_length  # recover the configured value

    def run(self, **kwds):
        """Convenience method to call the CoSimulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        self.simulation_length = kwds.pop("simulation_length", self.simulation_length)
        if self._cosimulation_flag:
            self._run_cosimulation(ts, xs, wall_time_start, **kwds)
        else:
            self._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=False, **kwds)
        for i in range(len(ts)):
            ts[i] = np.array(ts[i])
            xs[i] = np.array(xs[i])
        return list(zip(ts, xs))

    def interfaces_str(self):
        output = ""
        if self.output_interfaces or self.input_interfaces:
            output += 3 * LINE + "TVB <-> interfaces:\n\n"
            if self.output_interfaces:
                output += self.output_interfaces.print_str()
            if self.input_interfaces:
                output += self.input_interfaces.print_str()
            output += 2 * LINE
        return output

    def print_interfaces(self):
        print(self.interfaces_str())

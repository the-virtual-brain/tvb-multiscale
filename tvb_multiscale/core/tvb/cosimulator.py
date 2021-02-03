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

from tvb_multiscale.core.config import LINE
from tvb_multiscale.core.tvb.interfaces import TVBOutputInterfaces
from tvb_multiscale.core.tvb.interfaces import TVBInputInterfaces


class CoSimulator(CoSimulatorBase):

    tvb_output_interfaces = Attr(
        field_type=TVBOutputInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    tvb_input_interfaces = Attr(
        field_type=TVBInputInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    PRINT_PROGRESSION_MESSAGE = True

    def _configure_vois_and_proxy_inds(self):
        """This method will set the voi and proxy_inds of the CoSimulator,
            based on the predefined interfaces, which will also configure.
        """
        self.voi = []
        self.proxy_inds = []
        if self.tvb_output_interfaces:
            # Configure all TVB to Cosim interfaces:
            self.tvb_output_interfaces.configure()
            self.voi += self.tvb_output_interfaces.voi_unique
            self.proxy_inds += self.tvb_output_interfaces.proxy_inds_unique
        if self.tvb_input_interfaces:
            # Configure all Cosim to TVB interfaces:
            self.tvb_input_interfaces.configure(self)
            self.voi += self.tvb_input_interfaces.voi_unique
            self.proxy_inds += self.tvb_input_interfaces.proxy_inds_unique
        self.voi = np.unique(self.voi)
        self.proxy_inds = np.unique(self.proxy_inds)

    def _configure_local_vois_and_proxy_inds_per_interface(self):
        """This method will set the local -per cosimulation- voi and proxy_inds indices,
           based on the voi of each linked cosimulation monitor, for TVB to Cosimulator interfaces,
           and on the expected shape of ths cosimulation updates data for Cosimulator to TVB interfaces.
        """
        if self.tvb_output_interfaces:
            # Set the correct voi indices with reference to the linked TVB CosimMonitor, for each cosimulation:
            self.tvb_output_interfaces.set_local_indices(self.cosim_monitors)
        if self.tvb_input_interfaces:
            # Method to get the correct indices of voi and proxy_inds, for each cosimulation,
            # adjusted to the contents, shape etc of the cosim_updates,
            # based on TVB CoSmulators' vois and proxy_inds, i.e., good_cosim_update_values_shape
            self.tvb_input_interfaces.set_local_indices(self.voi, self.proxy_inds)

    def configure(self, full_configure=True):
        """Configure simulator and its components.

        The first step of configuration is to run the configure methods of all
        the Simulator's components, ie its traited attributes.

        Configuration of a Simulator primarily consists of calculating the
        attributes, etc, which depend on the combinations of the Simulator's
        traited attributes (keyword args).

        Converts delays from physical time units into integration steps
        and updates attributes that depend on combinations of the 6 inputs.

        Returns
        -------
        sim: Simulator
            The configured Simulator instance.

        """
        self._configure_vois_and_proxy_inds()
        super(CoSimulator, self).configure(full_configure=full_configure)
        self._configure_local_vois_and_proxy_inds_per_interface()
        return self

    def _prepare_stimulus(self):
        if self.simulation_length != self.synchronization_time:
            simulation_length = float(self.simulation_length)
            self.simulation_length = float(self.synchronization_time)
            super(CoSimulator, self)._prepare_stimulus()
            self.simulation_length = simulation_length
        else:
            super(CoSimulator, self)._prepare_stimulus()

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        if cosimulation and self.tvb_input_interfaces:
            # Get the update data from the other cosimulator
            cosim_updates = self.tvb_input_interfaces(self.good_cosim_update_values_shape)
            if np.all(np.isnan(cosim_updates)):
                cosim_updates = None
        else:
            cosim_updates = None
        # Loop of integration for synchronization_time
        current_step = int(self.current_step)
        for data in self(cosim_updates=cosim_updates, **kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        steps_performed = self.current_step - current_step
        if cosimulation and self.tvb_output_interfaces.interfaces:
            # Send the data to the other cosimulator
            self.tvb_output_interfaces.interfaces(
                self.loop_cosim_monitor_output(current_step, steps_performed))
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)
        return steps_performed

    def _run_cosimulation(self, ts, xs, wall_time_start, **kwds):
        simulation_time = 0.0
        remaining_time = self.simulation_length
        synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
        while remaining_time > 0:
            self.synchronization_n_step = np.minimum(synchronization_n_step,
                                                     int(remaining_time / self.integrator.dt))
            steps_performed = \
                self._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation=True, **kwds)
            simulation_time += steps_performed * self.integrator.dt
            remaining_time = self.simulation_length - simulation_time
            self.log.info("...%.3f%% completed!", simulation_time / self.simulation_length)
        self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
        if self._cosimulation_flag:
            # Run once more for synchronization steps in order to get the full delayed monitors' outputs:
            remaining_time = self.simulation_length + self.synchronization_time - simulation_time
            if remaining_time:
                self.log.info("Simulating for synchronization excess time %0.3f...", remaining_time)
                current_step = int(self.current_step)
                current_state = np.copy(self.current_state)
                synchronization_n_step = int(self.synchronization_n_step)  # store the configured value
                self.synchronization_n_step = np.minimum(synchronization_n_step,
                                                         int(remaining_time / self.integrator.dt))
                self._run_for_synchronization_time(ts, xs, wall_time_start,
                                                   cosimulation=False, **kwds)  # Run only TVB
                self.synchronization_n_step = int(synchronization_n_step)  # recover the configured value
                # Revert the current_step and current_state to those before the excess synchronization time
                self.current_step = int(current_step)
                self.current_state = np.copy(current_state)

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
        if self.tvb_output_interfaces or self.tvb_input_interfaces:
            output += 3 * LINE + "TVB <-> interfaces:\n\n"
            if self.tvb_output_interfaces:
                output += self.tvb_output_interfaces.print_str()
            if self.tvb_input_interfaces:
                output += self.tvb_input_interfaces.print_str()
            output += 2 * LINE
        return output

    def print_interfaces(self):
        print(self.interfaces_str())

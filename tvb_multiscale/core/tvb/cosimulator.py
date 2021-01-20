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
import math
import numpy as np

from tvb.basic.neotraits.api import Attr
from tvb.contrib.cosimulation.cosimulator import CoSimulator as CoSimulatorBase

from tvb_multiscale.core.tvb.interfaces.tvb_to_cosim_interfaces import TVBtoCosimInterfaces
from tvb_multiscale.core.tvb.interfaces.cosim_to_tvb_interfaces import CosimToTVBInterfaces


class CoSimulator(CoSimulatorBase):

    tvb_to_cosim_interfaces = Attr(
        field_type=TVBtoCosimInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""Interfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    cosim_to_tvb_interfaces = Attr(
        field_type=CosimToTVBInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""Interfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    PRINT_PROGRESSION_MESSAGE = True

    def _configure_vois_and_proxy_inds(self):
        """This method will set the voi and proxy_inds of the CoSimulator,
            based on the predefined interfaces, which will also configure.
        """
        self.voi = []
        self.proxy_inds = []
        if self.tvb_to_cosim_interfaces:
            # Configure all TVB to Cosim interfaces:
            self.tvb_to_cosim_interfaces.configure()
            self.voi += self.tvb_to_cosim_interfaces.voi_unique
            self.proxy_inds += self.tvb_to_cosim_interfaces.proxy_inds_unique
        if self.cosim_to_tvb_interfaces:
            # Configure all Cosim to TVB interfaces:
            self.cosim_to_tvb_interfaces.configure(self)
            self.voi += self.cosim_to_tvb_interfaces.voi_unique
            self.proxy_inds += self.cosim_to_tvb_interfaces.proxy_inds_unique
        self.voi = np.unique(self.voi)
        self.proxy_inds = np.unique(self.proxy_inds)

    def _configure_local_vois_and_proxy_inds_per_interface(self):
        """This method will set the local -per interface- voi and proxy_inds indices,
           based on the voi of each linked cosimulation monitor, for TVB to Cosimulator interfaces,
           and on the expected shape of ths cosimulation updates data for Cosimulator to TVB interfaces.
        """
        if self.tvb_to_cosim_interfaces:
            # Set the correct voi indices with reference to the linked TVB CosimMonitor, for each interface:
            self.tvb_to_cosim_interfaces.set_local_voi_indices(self.cosim_monitors)
        if self.cosim_to_tvb_interfaces:
            # Method to get the correct indices of voi and proxy_inds, for each interface,
            # adjusted to the contents, shape etc of the cosim_updates,
            # based on TVB CoSmulators' vois and proxy_inds, i.e., good_cosim_update_values_shape
            self.cosim_to_tvb_interfaces.set_local_voi_and_proxy_indices(self.voi, self.proxy_inds)

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

    def _run_for_synchronization_time(self, step_synch, ts, xs, wall_time_start, interface=True, **kwds):
        if self.cosim_to_tvb_interfaces:
            # Get the update data from the other cosimulator
            cosim_updates = self.cosim_to_tvb_interfaces(self.good_cosim_update_values_shape)
            if np.all(np.isnan(cosim_updates)):
                cosim_updates = None
        else:
            cosim_updates = None
        # Loop of integration for synchronization_time
        for data in self(cosim_updates=cosim_updates, **kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        if self.tvb_to_cosim_interfaces.interfaces and interface:
            # Send the data to the other cosimulator
            self.tvb_to_cosim_interfaces.interfaces(
                self.loop_cosim_monitor_output(step_synch * self.synchronization_n_step,
                                               self.synchronization_n_step))
        elapsed_wall_time = time.time() - wall_time_start
        self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                      elapsed_wall_time * 1e3 / self.simulation_length)

    def run(self, **kwds):
        """Convenience method to call the CoSimulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        self.simulation_length = kwds.pop("simulation_length", self.simulation_length)
        simulation_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
        if self._cosimulation_flag:
            loop_n_step = self.synchronization_n_step
        else:
            loop_n_step = simulation_steps  # loop only once
        for step_synch in range(0, simulation_steps, loop_n_step):
            self._run_for_synchronization_time(step_synch, ts, xs, wall_time_start, interface=True, **kwds)
        if self._cosimulation_flag:
            current_step = int(self.current_step)
            current_state = np.copy(self.current_state)
            # Run once more for synchronization steps in order to get the full delayed monitors' outputs,
            # but without sending data to the other cosimulator:
            self._run_for_synchronization_time(step_synch, ts, xs, wall_time_start, interface=False, **kwds)
            # Revert the current_step and current_state to those before the excess step_synch time
            self.current_step = int(current_step)
            self.current_state = np.copy(current_state)
        for i in range(len(ts)):
            ts[i] = np.array(ts[i])
            xs[i] = np.array(xs[i])
        return list(zip(ts, xs))

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

    def _configure_cosimulation(self):
        """This method will set the voi and proxy_inds of the CoSimulator,
            based on the predefined interfaces, which will also configure.
        """
        self.voi = []
        self.proxy_inds = []
        if self.tvb_to_cosim_interfaces:
            # Configure all TVB to Cosim interfaces:
            self.tvb_to_cosim_interfaces.configure()
        if self.cosim_to_tvb_interfaces:
            # Configure all Cosim to TVB interfaces:
            self.cosim_to_tvb_interfaces.configure(self)
            for voi in self.cosim_to_tvb_interfaces.voi:
                self.voi += voi.tolist()
            for interface in self.cosim_to_tvb_interfaces.interfaces:
                for proxy_ind in interface.proxy_inds:
                    self.proxy_inds += proxy_ind.tolist()
        self.voi = np.unique(self.voi)
        self.proxy_inds = np.unique(self.proxy_inds)

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
        self._configure_cosimulation()
        super(CoSimulator, self).configure(full_configure=full_configure)
        return self

    def _print_progression_message(self, step, n_steps):
        """
        #TODO do yu it for the moment
        :param step:
        :param n_steps:
        :return:
        """
        if step - self.current_step >= self._tic_point:
            toc = time.time() - self._tic
            if toc > 600:
                if toc > 7200:
                    time_string = "%0.1f hours" % (toc / 3600)
                else:
                    time_string = "%0.1f min" % (toc / 60)
            else:
                time_string = "%0.1f sec" % toc
            print_this = "\r...%0.1f%% done in %s" % \
                         (100.0 * (step - self.current_step) / n_steps, time_string)
            self.log.info(print_this)
            self._tic_point += self._tic_ratio * n_steps

    def run(self, **kwds):
        # TODO need to be test it
        """Convenience method to call the simulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        # Loop over all synchronization steps
        for step_synch in range(0, int(math.ceil(self.simulation_length / self.integrator.dt)),
                                self.synchronization_n_step):
            if self.cosim_to_tvb_interfaces:
                # Get the update data from the other cosimulator
                cosim_updates = self.cosim_to_tvb_interfaces(self.good_cosim_update_values_shape)
            else:
                cosim_updates = None
            # Loop of integration for synchronization_time
            for data in self(cosim_updates=cosim_updates, **kwds):
                for tl, xl, t_x in zip(ts, xs, data):
                    if t_x is not None:
                        t, x = t_x
                        tl.append(t)
                        xl.append(x)
            if self.tvb_to_cosim_interfaces.interfaces:
                # Send the data to the other cosimulator
                self.tvb_to_cosim_interfaces.interfaces(
                    self.loop_cosim_monitor_output(step_synch*self.synchronization_n_step,
                                                               self.synchronization_n_step))
            elapsed_wall_time = time.time() - wall_time_start
            self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                          elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = np.array(ts[i])
            xs[i] = np.array(xs[i])
        return list(zip(ts, xs))

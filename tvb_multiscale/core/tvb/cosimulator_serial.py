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

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class CoSimulatorSerial(CoSimulator):

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""A SpikingNetwork class to simulate the spiking nework simulator.""",
        required=True,
        default=None
    )

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
        super(CoSimulatorSerial, self).configure(full_configure=full_configure)
        if self.spiking_network:
            self.spiking_network.configure()

    def _run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True, **kwds):
        steps_performed = \
            super(CoSimulatorSerial, self)._run_for_synchronization_time(ts, xs, wall_time_start, cosimulation, **kwds)
        if cosimulation:
            if self.spiking_network:
                self.log.info("Simulating the spikning network for %d time steps...", steps_performed)
                self.spiking_network.Run(steps_performed)
        return steps_performed

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

import numpy

from tvb.basic.neotraits.api import Int
from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator


class CoSimulatorRemoteParallel(CoSimulator):

    min_idelay_synch_n_step_ratio = Int(
        label="min_idelay_synch_n_step_ratio",
        choices=(1, 2),
        default=2,
        required=True,
        doc="""min_idelay to synchronization_n_step ratio, 
               i.e., an integer value defining how many times smaller should the synchronization time be 
               compared to the minimum delay time in integration time steps.
               For the moment we limit it to 1 (synchronization_time = min_delay) 
               or 2 (synchronization_time = min_delay/2)""")

    def run_for_synchronization_time(self, ts, xs, wall_time_start, cosimulation=True):
        self.n_tvb_steps_ran_since_last_synch = \
            super(CoSimulatorRemoteParallel, self).run_for_synchronization_time(
                ts, xs, wall_time_start, cosim_updates=self.get_cosim_updates(cosimulation))
        return self.send_cosim_coupling(cosimulation)


class CoSimulatorParallel(CoSimulatorRemoteParallel):

    def get_cosim_updates(self, cosim_updates=None, cosimulation=True):
        if cosimulation and self.input_interfaces and cosim_updates is not None:
            # Get the update data from the other cosimulator
            cosim_updates = self.input_interfaces(cosim_updates, self.good_cosim_update_values_shape)
            isnans = numpy.isnan(cosim_updates[-1])
            if numpy.all(isnans):
                cosim_updates = None
                self.log.warning("No or all NaN valued cosimulator updates at time step %d!" % self.current_step)
            elif numpy.any(isnans):
                msg = "NaN values detected in cosimulator updates at time step %d!" % self.current_step
                self.log.error(msg)
                raise Exception(msg)
        return cosim_updates

    def run_for_synchronization_time(self, ts, xs, wall_time_start, cosim_updates=None, cosimulation=True):
        self.n_tvb_steps_ran_since_last_synch = \
            super(CoSimulatorRemoteParallel, self).run_for_synchronization_time(
                ts, xs, wall_time_start, cosim_updates=self.get_cosim_updates(cosim_updates, cosimulation))
        return self.send_cosim_coupling(cosimulation)


class CoSimulatorParallelNRP(CoSimulatorParallel):

    pass

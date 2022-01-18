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

"""
Jansen-Rit and derivative models.

"""

import numpy

from tvb.basic.neotraits.api import Int
from tvb.simulator.models.jansen_rit import JansenRit as JansenRitTVB
from tvb.simulator.coupling import SigmoidalJansenRit as SigmoidalJansenRitTVB


class JansenRit(JansenRitTVB):

    cvar = numpy.array([0, 1, 2], dtype=numpy.int32)


class SigmoidalJansenRit(SigmoidalJansenRitTVB):
    r"""
    Provides a sigmoidal coupling function as described in the
    Jansen and Rit model, of the following form

    .. math::
        c_{min} + (c_{max} - c_{min}) / (1.0 + \exp(-a(x-midpoint)/\sigma))

    Assumes that x has have three state variables.

    """

    Ein = Int(default=1,
              field_type=int,
              label="Excitatory interneurons Index",
              doc="""Index of excitatory interneurons coupling state variable""")

    Iin = Int(default=2,
              field_type=int,
              label="Inhibitory interneurons Index",
              doc="""Index of inhibitory interneurons coupling state variable""")

    def pre(self, x_i, x_j):
        pre = self.cmax / (1.0 + numpy.exp(self.r * (self.midpoint - (x_j[:, self.Ein] - x_j[:, self.Iin]))))
        return pre[:, numpy.newaxis]

# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Models based on Wong-Wang's work.
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""

from numba import guvectorize, float64
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.base import numpy
from tvb.basic.neotraits.api import NArray, Range


@guvectorize([(float64[:],)*5], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
def _numba_dfun(S, R, g, t, dx):
    "Gufunc for linear reduced Wong-Wang model equations."
    dx[0] = - (S[0] / t[0]) + R[0] * g[0]   # S


class LinearReducedWongWangExcIO(ReducedWongWangExcIO):

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.0, hi=0.200, step=0.001),
        doc="""[s]. Parameter chosen to fit numerical solutions.""")

    def _numpy_dfun(self, integration_variables, R):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                  x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

        """

        S = integration_variables[0]  # Synaptic gating dynamics

        # Synaptic gating dynamics
        dS = - (S / self.tau_s) + R * self.gamma

        return numpy.array([dS])

    def dfun(self, x, c, local_coupling=0.0, time=0.0):
        if self._R is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R = state_variables[1]  # Rates
        else:
            R = self._R
        if self.use_numba:
            deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R,  # Variables (n_regions, n_svs)
                                self.gamma, self.tau_s)        # Parameters (n_regions, )
            deriv = deriv.T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, R)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        return deriv


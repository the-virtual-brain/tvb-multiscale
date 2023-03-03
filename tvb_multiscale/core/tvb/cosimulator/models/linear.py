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
Generic linear model.

"""

import numpy
from numba import guvectorize, float64
from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, Final, List, Range


# @guvectorize([(float64[:],)*8], '(n),(m)(k)' + ',()'*4 + '->(n)', nopython=True)  # lc_0 as variable
@guvectorize([(float64[:],)*8], '(n),(m)' + ',()'*5 + '->(n)', nopython=True)  # lc_0 as parameter
def _numba_dfun(R, c, lc, t, gm, io, g, dR):
    "Gufunc for Linear model equations."
    dR[0] = (gm[0] * R[0] + g[0] * c[0] + lc[0]) / t[0] + io[0]


class Linear(Model):

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([100.0]),
        domain=Range(lo=-0.1, hi=100.0, step=0.1),
        doc="Time constant")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([-1.0]),
        domain=Range(lo=-100.0, hi=0.0, step=1.0),
        doc="The damping coefficient specifies how quickly the node's activity relaxes, must be larger"
            " than the node's in-degree in order to remain stable.")

    I_o = NArray(
        label=r":math:`I_o`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="External stimulus")

    G = NArray(
        label=r":math:`G`",
        default=numpy.array([2.0]),
        domain=Range(lo=-0.0, hi=100.0, step=1.0),
        doc="Global coupling scaling")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"R": numpy.array([0, 100.0])},
        doc="Range used for state variable initialization and visualization.")

    state_variable_boundaries = Final(
        default={"R": numpy.array([0.0, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
                the boundaries of the dynamic range of that state-variable. 
                Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("R",),
        default=("R",), )

    state_variables = ('R',)
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)
    use_numba = True

    def _numpy_fun(self, state, coupling, local_coupling=0.0):
        R, = state
        c, = coupling
        dR = (self.gamma * R + self.G * c + local_coupling * R) / self.tau + self.I_o
        return numpy.array([dR])

    def dfun(self, state, coupling, local_coupling=0.0, time=0.0):
        """
        .. math::
            dR = ({\gamma}R + G * coupling + local_coupling * R)/{\tau} + I_o
        """
        if self.use_numba:
            # Variables:
            R_ = state.reshape(state.shape[:-1]).T
            c_ = coupling.reshape(coupling.shape[:-1]).T
            lc_0 = local_coupling * state[0]
            deriv = _numba_dfun(R_, c_,
                                lc_0[:, 0],  # as parameter, or lc_0 as variable
                                self.tau, self.gamma, self.I_o, self.G)
            return deriv.T[..., numpy.newaxis]
        else:
            return self._numpy_fun(state, coupling, local_coupling)


# @guvectorize([(float64[:],)*5], '(n),(m)(k)' + ',()' + '->(n)', nopython=True)  # lc_0 as variable
@guvectorize([(float64[:],)*5], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)   # lc_0 as parameter
def _numba_update_non_state_variables_before_integration(R, c, lc_0, g, newR):
    "Gufunc for reduced Wong-Wang model equations."
    newR[0] = R[0]  # R
    newR[1] = g[0]*c[0] + lc_0[0]*R[0]  # Rin


# @guvectorize([(float64[:],)*6], '(n)' + ',()'*4 + '->(n)', nopython=True)  # Rin as parameter
@guvectorize([(float64[:],)*6], '(n),(m)' + ',()'*3 + '->(n)', nopython=True)  # Rin as variable
def _numba_dfun_Rin(R, rin, t, gm, io, dR):
    "Gufunc for Linear model equations."
    dR[0] = (gm[0] * R[0] + rin[0]) / t[0] + io[0]


class LinearRin(Linear):

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"R": numpy.array([0, 100.0]),
                 "Rin": numpy.array([-100.0, 100.0])},
        doc="Range used for state variable initialization and visualization.")

    state_variable_boundaries = Final(
        default={"R": numpy.array([0.0, None]),
                 "Rin": numpy.array([None, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
                    the boundaries of the dynamic range of that state-variable. 
                    Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("R", " Rin"),
        default=("R", " Rin"), )

    state_variables = ('R', "Rin")
    _nvar = 2
    non_integrated_variables = ['Rin']

    _Rin = None
    _stimulus = 0.0

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self.n_nonintvar = self.nvar - self.nintvar
        self._Rin = None
        self._stimulus = 0.0

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            # Variables (n_regions, n_svs):
            sv_ = state_variables.reshape(state_variables.shape[:-1]).T
            c_ = coupling.reshape(coupling.shape[:-1]).T
            # Parameters (n_regions, )
            lc_0 = numpy.array([local_coupling, ])  # as parameter, or local_coupling *  numpy.ones((1,1)) as variable
            state_variables = \
                _numba_update_non_state_variables_before_integration(sv_, c_,       # variables
                                                                     lc_0, self.G)  # parameters
            state_variables = state_variables.T[..., numpy.newaxis]
        else:
            R = state_variables[0]  # synaptic gating dynamics

            c_0 = coupling[0]

            # if applicable
            lc_0 = local_coupling * R[0]

            Rin = self.G * (c_0 + lc_0)

            state_variables[1] = Rin

        # Keep them here so that they are not recomputed in the dfun
        self._Rin = numpy.copy(state_variables[1])
        return state_variables

    def _numpy_dfun(self, integration_variables, Rin):
        R = integration_variables[0]
        dR = (self.gamma * R + Rin) / self.tau + self.I_o
        return numpy.array([dR])

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

    def dfun(self, x, c, local_coupling=0.0):
        if self._Rin is None:
            state_variables = self._integration_to_state_variables(x)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            Rin = state_variables[1]  # Input Rates
        else:
            Rin = self._Rin
        if self.use_numba:
            # Variables
            deriv = _numba_dfun_Rin(x.reshape(x.shape[:-1]).T,  # variables (n_regions, n_svs)
                                    Rin,  # Rin for as a variable, Rin[:, 0] for as a parameter (n_regions, )
                                    self.tau, self.gamma, self.I_o)  # parameters
            deriv = deriv.T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(x, Rin)
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._Rin = None
        return deriv

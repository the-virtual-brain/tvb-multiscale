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
from tvb.simulator.models.wong_wang import ReducedWongWang as TVBReducedWongWang
from tvb.simulator.models.base import numpy
from tvb.basic.neotraits.api import NArray, Final, List, Range


@guvectorize([(float64[:],)*10], '(n),(m)' + ',()'*7 + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration(S, c, a, b, d, w, jn, g, io, newS):
    "Gufunc for reduced Wong-Wang model equations."
    newS[0] = S[0]  # S
    cc = g[0]*jn[0]*c[0]
    I = w[0] * jn[0] * S[0] + io[0] + cc  # I
    x = a[0]*I - b[0]
    h = x / (1 - numpy.exp(-d[0]*x))
    newS[1] = h     # R


@guvectorize([(float64[:],)*5], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
def _numba_dfun(S, R, g, t, dx):
    "Gufunc for reduced Wong-Wang model equations."
    dx[0] = - (S[0] / t[0]) + (1.0 - S[0]) * R[0] * g[0]   # S


class ReducedWongWangExcIO(TVBReducedWongWang):

    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini,
                  Gian Luca Romani, Patric Hagmann, and Maurizio Corbetta.
                  *Resting-State Functional Connectivity Emerges from
                  Structurally and Dynamically Shaped Slow Linear Fluctuations*.
                  The Journal of Neuroscience 33(27), 11239 –11252, 2013.



    .. automethod:: ReducedWongWangExcIO.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

    """

    # Define traited attributes for this model, these represent possible kwargs.

    a = NArray(
        label=":math:`a`",
        default=numpy.array([270., ]),
        domain=Range(lo=0.0, hi=0.270, step=0.01),
        doc="[n/C]. Input gain parameter, chosen to fit numerical solutions.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([108., ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="[Hz]. Input shift parameter chosen to fit numerical solutions.")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.154, ]),
        domain=Range(lo=0.0, hi=0.200, step=0.001),
        doc="""[s]. Parameter chosen to fit numerical solutions.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
        doc="""Kinetic parameter""")

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        domain=Range(lo=1.0, hi=150.0, step=1.0),
        doc="""[ms]. NMDA decay time constant.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.9, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence""")

    J_N = NArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2609, ]),
        domain=Range(lo=0.0000, hi=0.5, step=0.0001),
        doc="""Excitatory recurrence""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.3, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    sigma_noise = NArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        domain=Range(lo=0.0, hi=0.005, step=0.0001),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
            integration schemes.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "R": numpy.array([0.0, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    state_variable_range = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "R": numpy.array([0.0, 1000.0])},
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S', 'R'),
        default=('S', 'R'),
        doc="""default state variables to be monitored""")

    state_variables = ['S', 'R']
    non_integrated_variables = ['R']
    _nvar = 2
    cvar = numpy.array([0, 1], dtype=numpy.int32)
    _R = None
    _stimulus = 0.0
    use_numba = True

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self.n_nonintvar = self.nvar - self.nintvar
        self._R = None
        self._stimulus = 0.0

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration(
                    # Variables (n_regions, n_svs):
                    state_variables.reshape(state_variables.shape[:-1]).T,
                    coupling.reshape(coupling.shape[:-1]).T + local_coupling * state_variables[0],
                    # Parameters (n_regions, ):
                    self.a, self.b, self.d, self.w, self.J_N, self.G, self.I_o)
            state_variables = state_variables.T[..., numpy.newaxis]

        else:
            S = state_variables[0]  # synaptic gating dynamics

            c_0 = coupling[0]

            # if applicable
            lc_0 = local_coupling * S[0]

            coupling = self.G * self.J_N * (c_0 + lc_0)

            # Currents
            I = self.w * self.J_N * S + self.I_o + coupling
            x = self.a * I - self.b

            # Rates
            R = x / (1 - numpy.exp(-self.d * x))

            # We now update the state_variable vector with the new rate:
            state_variables[1] = R

        # Keep them here so that they are not recomputed in the dfun
        self._R = numpy.copy(state_variables[1])

        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

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
        dS = - (S / self.tau_s) + (1 - S) * R * self.gamma

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


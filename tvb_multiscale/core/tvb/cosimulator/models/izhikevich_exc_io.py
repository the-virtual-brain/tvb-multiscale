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
Models based on Izhikevich's work.
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>
.. moduleauthor:: André Blickensdörfer <andre.blickensdoerfer@charite.de>
"""

# from numba import guvectorize, float64
from tvb.simulator.models.base import numpy, ModelNumbaDfun
from tvb.basic.neotraits.api import NArray, Final, List, Range


# @guvectorize([(float64[:],)*15], '(n),(m)' + ',()'*12 + '->(n)', nopython=True)
# def _numba_update_non_state_variables_before_integration(X, cc,
#                                                          n0, n1, n2, c, omega, fmin,
#                                                          v_th, E_rev, g_max, w, G, I_ext, newX):
#     "Gufunc for reduced Izhikevich model equations."
#     newX[0] = X[0]  # S
#     newX[1] = X[1]  # V
#     newX[2] = X[2]  # U
#
#     # Total current = Iext - Synaptic current
#     # = Iext -  (w*S + G*coupling) * g * (V - E_rev)
#     I = I_ext - (w[0] * X[0] + G[0] * cc[0]) * g_max[0] * (X[1] - E_rev[0])
#
#     # Sigmoidal synaptic function:
#     H = 1.0 / (1 + numpy.exp(-omega[0] * (X[1] - (c[0]))))
#
#     J = I - X[2]  # = I - U
#     nc = n2[0] * c[0] * c[0] + n1[0] * c[0] + n0[0] + J
#
#     # Rate computation:
#     R = fmin[0]
#     if nc > 0.0:
#         sqrtJj =  numpy.sqrt(4 * (J + 0j) - 65)
#         Tv = (10/sqrtJj) * numpy.arctan(((2 * v_th[0]) + 125) / (5 * sqrtJj))
#         Tc = (10 / sqrtJj) * numpy.arctan(((2 * c[0]) + 125) / (5 * sqrtJj))
#         f = 1.0 / (Tv - Tc)
#         if f > R:
#             R = f
#
#     newX[3] = H * R # R
#     newX[4] = I     # I
#
#
# @guvectorize([(float64[:],)*15], '(n),(m)' + ',()'*11 + '->(n)', nopython=True)
# def _numba_dfun(X, R, I, tau_s, n0, n1, n2, a, b, c, d, kappa, ckappa, dnck, dx):
#     "Gufunc for reduced Izhikevich model equations."
#
#     # Synaptic gating dynamics
#     # dS = R - S / tau_s
#     dx[0] = R[0] - X[0] / tau_s[0]
#
#     # Membrane potential dynamics
#     J = I[0] - X[2]  # = I - U
#     #     =  nfun(c) - nfun(c + kappa)
#     ncknc =  n2[0] * (ckappa[0] * ckappa[0] - c[0] * c[0]) +  n1[0] * (ckappa[0] - c[0])
#     eta = dnck[0] / ncknc
#     xi = ncknc / numpy.exp(eta * kappa[0])
#     dx[1]  = n2[0] * X[1] * X[1] + n1[0] * X[1] + n0[0] + J - xi * numpy.exp(eta * (X[1] - c[0]))
#
#     # Recovery dynamics
#     # dU = a * (bV - U) + dR
#     dx[2]  = a[0] * (b[0] * X[1]  - X[2] ) + R[0] * d[0]


class IzhikevichExcIO(ModelNumbaDfun):

    r"""
    .. [VG_2014] Sid Visser and Stephan A Van Gils,
                 *Lumping Izhikevich neurons*.
                 EPJ Nonlinear Biomedical Physics 2(6), 2014.



    .. automethod:: IzhikevichExcIO.__init__

    Equations taken from [[VG_2014]_

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

    """

    # Define traited attributes for this model, these represent possible kwargs.

    n0 = NArray(
        label=":math:`n_0`",
        default=numpy.array([140.0, ]),
        domain=Range(lo=0.001, hi=150.0, step=0.001),
        doc="[?]. Constant term of polynomial.")

    n1 = NArray(
        label=":math:`n_1`",
        default=numpy.array([5.0, ]),
        domain=Range(lo=1.0, hi=10.0, step=0.001),
        doc="[?]. Linear term of polynomial.")

    n2 = NArray(
        label=":math:`n_2`",
        default=numpy.array([0.04, ]),
        domain=Range(lo=0.01, hi=0.5, step=0.001),
        doc="[?]. Square term of polynomial.")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.02, ]),
        domain=Range(lo=0.001, hi=0.1, step=0.001),
        doc="[?]. Decay rate of recovery variable.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.05),
        doc="[?]. Slope parameter of u nullcline.")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([-55.0, ]),
        domain=Range(lo=-100.0, hi=0.0, step=5.0),
        doc="""[mV]. Reset value after spike 
                for the original Izhikevich spiking neuronal model.""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.05, hi=8.0, step=0.05),
        doc="""[?]. Increment of u after spike
                for the original Izhikevich spiking neuronal model.""")

    fmin = NArray(
        label=":math:`f_{min}`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.1),
        doc="""[Hz]. ??Minimum firing rate.""")

    kappa = NArray(
        label=":math:`\kappa`",
        default=numpy.array([0.8, ]),
        domain=Range(lo=0.5, hi=2.0, step=0.1),
        doc="""[?]. ??.""")

    omega = NArray(
        label=":math:`\omega`",
        default=numpy.array([20.0, ]),
        domain=Range(lo=1.0, hi=50.0, step=1.0),
        doc="""[real?]. Steepness of sigmoidal.""")

    v_th = NArray(
        label=":math:`v_{th}`",
        default=numpy.array([30.0, ]),
        domain=Range(lo=0.0, hi=100.0, step=5.0),
        doc="""[mV]. Membrane potential threshold value for spiking 
               for the original Izhikevich spiking neuronal model.""")

    E_rev = NArray(
        label=":math:`E_{rev}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=100.0, step=5.0),
        doc="""[mV]. Reversal synaptic potential value .""")

    g_max = NArray(
        label=r":math:`\g_{max}`",
        default=numpy.array([0.002, ]),
        domain=Range(lo=0.0, hi=0.1, step=0.001),
        doc="""[nS] Maximum conductance""")

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=numpy.array([10., ]),
        domain=Range(lo=1.0, hi=100.0, step=1.0),
        doc="""[ms]. Synaptic time constant.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.1, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence weight.""")

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=-100.0, hi=100.0, step=0.1),
        doc="""[pA] Constant applied background current.""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "V": numpy.array([None, None]),
                 "U": numpy.array([None, None]),
                 "R": numpy.array([0.0, None]),
                 "I": numpy.array([None, None])
                 },
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")

    state_variable_range = Final(
        default={"S": numpy.array([0.0, 1.0]),
                 "V": numpy.array([-80.0, 30.0]),
                 "U": numpy.array([-15.0, 5.0]),
                 "R": numpy.array([0.0, 100.0]),
                 "I": numpy.array([-10.0, 10.0]),
                 },
        label="State variable ranges [lo, hi]",
        doc="Initial condition ranges for the state variables")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S', 'V', 'U', 'R', 'I'),
        default=('S', 'V', 'U', 'R', 'I'),
        doc="""default state variables to be monitored""")

    state_variables = ['S', 'V', 'U', 'R', 'I']
    non_integrated_variables = ['R', 'I']
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)  # [state_variables_indices], in our case this is "S", i.e., indice 0
    # Parameters used to store intermediate variables and stimulus between calls
    _R = None
    _I = None
    _J = None
    _nc = None
    _stimulus = 0.0

    use_numba = False

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self.n_nonintvar = self.nvar - self.nintvar # number of non integrated state variables
        self._R = None
        self._I = None  # = I_ext - I_syn
        self._J = None  # = I - U
        self._nc = None
        self._ckappa = self.c + self.kappa  # c + kappa
        self._dnck = 2 * self.n2 * self._ckappa + self.n1  # derivative of _nfun at point c + kappa
        self._stimulus = 0.0

    def _nfun(self, x):
        """This method computes the polynomial term of the dV/dt function"""
        return self.n2 * x * x + self.n1 * x + self.n0 + self._J

    def _T(self, sqrtJj, v):
        _T = (10 / sqrtJj) * numpy.arctan(((2 * v) + 125) / (5 * sqrtJj))
        # print("\n_T = \n", _T)
        return _T

    def _f(self):
        sqrtJj = numpy.sqrt(4 * (self._J + 0j) - 65)
        # print("\nsqrtJj = \n", sqrtJj)
        _f = numpy.maximum(1.0/(self._T(sqrtJj, self.v_th) - self._T(sqrtJj, self.c)), self.fmin)
        # print("\n_f = \n", _f)
        return _f

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        """This method computes intermediate -non integrated- state variables
           to be used for the computation of the dfun of the dynamical - i.e., integrated - state variables"""
        self._stimulus = stimulus
        # if self.use_numba:
        #     state_variables = \
        #         _numba_update_non_state_variables_before_integration(
        #             state_variables.reshape(state_variables.shape[:-1]).T,
        #             coupling.reshape(coupling.shape[:-1]).T +
        #             local_coupling * state_variables[0],
        #             self.n0, self.n1, self.n2, self.c, self.omega, self.fmin,
        #             self.v_th, self.E_rev, self.g_max, self.w, self.G, self.I_ext)
        #     return state_variables.T[..., numpy.newaxis]

        # state_variables shape: (var, brain regions, modes)
        S = state_variables[0, :]  # synaptic gating dynamics, shape: (brain regions, modes)
        V = state_variables[1, :]  # membrane potential, shape: (brain regions, modes)
        U = state_variables[2, :]  # recovery variable, shape: (brain regions, modes)

        # coupling is large scale coupling coming from other brain regions, state_variables[cvar], see cvar above:
        # coupling shape: (cvars, brain regions, modes)
        c_0 = coupling[0, :]  # c_0 shape: (brain regions, modes) = sum(w_ij * S_j) for linear additive coupling

        # local coupling for surface simulations, if applicable
        lc_0 = local_coupling * S

        # Total current = Iext - Synaptic current
        # = Iext -  (w*S + G*coupling) * g * (V - E_rev)
        self._I = self.I_ext - (self.w * S + self.G * (c_0 + lc_0)) * self.g_max * (V - self.E_rev)

        # Sigmoidal synaptic function:
        H = 1.0 / (1 + numpy.exp(-self.omega * (V - (self.c))))
        # print("\nH = \n", H)

        # These two will be used later as well for dfun:
        self._J = self._I - U
        self._nc = self._nfun(self.c)

        # Rate computation:
        self._R = numpy.real(H * numpy.where(self._nc > 0.0, self._f(), self.fmin))
        # print("\n_R = \n", self._R)
        # We now update the state_variables vector with the new rate and current:
        state_variables[3, :] = self._R.copy()
        state_variables[4, :] = self._I.copy()

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
        # integration_variables shape: (var, brain regions, modes)
        S = integration_variables[0, :]  # synaptic gating dynamics, shape: (brain regions, modes)
        V = integration_variables[1, :]  # membrane potential, shape: (brain regions, modes)
        U = integration_variables[2, :]  # recovery variable, shape: (brain regions, modes)

        # Membrane potential dynamics
        ncknc = self._nfun(self._ckappa) - self._nc
        eta = self._dnck / ncknc
        xi = ncknc / numpy.exp(eta * self.kappa)
        dV = self._nfun(V) - xi * numpy.exp(eta * (V - self.c))

        # Recovery dynamics
        dU = self.a*(self.b * V - U) + R * self.d

        # Synaptic gating dynamics
        dS = R - S / self.tau_s

        return numpy.array([dS, dV, dU])

    def dfun(self, x, c, local_coupling=0.0):  # x = [S, V, U]
        if self._R is None:
            # Compute intermediate values for this step
            # Form the whole state_variables vector from integration_variables (i.e., x)
            state_variables = self._integration_to_state_variables(x)  # [S, V, U] + [R, I]
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R = state_variables[3]  # Rates
            # I = state_variables[4]  # Currents
        else:
            # Just set the intermediate values necessary for dfun
            R = self._R
            # I = self._I
        # if self.use_numba:
        #     deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R.reshape(x.shape[:-1]).T, I.reshape(x.shape[:-1]).T,
        #                         self.tau_s, self.n0, self.n1, self.n2, self.a, self.b, self.c, self.d,
        #                         self.kappa, self._ckappa, self._dnck)
        #     deriv = deriv.T[..., numpy.newaxis]
        # else:
        deriv = self._numpy_dfun(x, R)  # x = [S, V, U]
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R = None
        self._I = None
        self._J = None
        self._nc = None
        return deriv


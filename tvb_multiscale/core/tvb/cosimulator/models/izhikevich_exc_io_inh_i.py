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


# @guvectorize([(float64[:],)*18], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
# def _numba_update_non_state_variables_before_integration(S, c,
#                                                          ae, be, de, wp, we, jn,
#                                                          ai, bi, di, wi, ji,
#                                                          g, l, io, ie, newS):
#     "Gufunc for reduced Wong-Wang model equations."
#
#     newS[0] = S[0]  # S_e
#     newS[1] = S[1]  # S_i
#
#     cc = g[0]*jn[0]*c[0]
#
#     jnSe = jn[0] * S[0]
#
#     I_e = wp[0] * jnSe - ji[0] * S[1] + we[0] * io[0] + cc + ie[0]  # I_e
#     x = ae[0]*I_e - be[0]
#     h = x / (1 - numpy.exp(-de[0]*x))
#     newS[2] = h  # R_e
#
#     I_i = jnSe - S[1] + wi[0] * io[0] + l[0] * cc   # I_i
#     x = ai[0]*I_i - bi[0]
#     h = x / (1 - numpy.exp(-di[0]*x))
#     newS[3] = h  # R_i
#
#
# @guvectorize([(float64[:],)*7], '(n),(m)' + ',()'*4 + '->(n)', nopython=True)
# def _numba_dfun(S, R, ge, te, gi, ti, dx):
#     "Gufunc for reduced Wong-Wang model equations."
#     dx[0] = - (S[0] / te[0]) + (1.0 - S[0]) * R[0] * ge[0]
#     dx[1] = - (S[1] / ti[0]) + R[1] * gi[0]


class IzhikevichExcIOInhI(ModelNumbaDfun):
    r"""
    TODO: update equations in simple exc version, then copy here as well
    """

    # Define traited attributes for this model, these represent possible kwargs.
    # TODO: check units and update in exc only version as well

    # Attributes of excitatory population.
    ne0 = NArray(
        label=":math:`n_e_0`",
        default=numpy.array([140.0, ]),
        domain=Range(lo=0.001, hi=150.0, step=0.001),
        doc="[?]. Constant term of polynomial in excitatory population.")

    ne1 = NArray(
        label=":math:`n_e_1`",
        default=numpy.array([5.0, ]),
        domain=Range(lo=1.0, hi=10.0, step=0.001),
        doc="[?]. Linear term of polynomial in excitatory population.")

    ne2 = NArray(
        label=":math:`n_e_2`",
        default=numpy.array([0.04, ]),
        domain=Range(lo=0.01, hi=0.5, step=0.001),
        doc="[?]. Square term of polynomial in excitatory population.")

    ae = NArray(
        label=":math:`a_e`",
        default=numpy.array([0.02, ]),
        domain=Range(lo=0.001, hi=0.1, step=0.001),
        doc="[?]. Decay rate of recovery variable in excitatory population.")

    be = NArray(
        label=":math:`b_e`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.05),
        doc="[?]. Slope parameter of u nullcline in excitatory population.")

    ce = NArray(
        label=":math:`c_e`",
        default=numpy.array([-55.0, ]),
        domain=Range(lo=-100.0, hi=0.0, step=5.0),
        doc="""[mV]. For excitatory population. Reset value after spike 
                for the original Izhikevich spiking neuronal model.""")

    de = NArray(
        label=":math:`d_e`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.05, hi=8.0, step=0.05),
        doc="""[?]. For excitatory population. Increment of u after spike
                for the original Izhikevich spiking neuronal model.""")

    fmine = NArray(
        label=":math:`f_{min}_e`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.1),
        doc="""[Hz]. ??Minimum firing rate in excitatory population.""")

    v_the = NArray(
        label=":math:`v_{th}_e`",
        default=numpy.array([30.0, ]),
        domain=Range(lo=0.0, hi=100.0, step=5.0),
        doc="""[mV]. For excitatory population. Membrane potential threshold 
                value for spiking for the original Izhikevich spiking neuronal model.""")

    # Attributes of inhibitory population.
    ni0 = NArray(
        label=":math:`n_i_0`",
        default=numpy.array([140.0, ]),
        domain=Range(lo=0.001, hi=150.0, step=0.001),
        doc="[?]. Constant term of polynomial in inhibitory population.")

    ni1 = NArray(
        label=":math:`n_i_1`",
        default=numpy.array([5.0, ]),
        domain=Range(lo=1.0, hi=10.0, step=0.001),
        doc="[?]. Linear term of polynomial in inhibitory population.")

    ni2 = NArray(
        label=":math:`n_i_2`",
        default=numpy.array([0.04, ]),
        domain=Range(lo=0.01, hi=0.5, step=0.001),
        doc="[?]. Square term of polynomial in inhibitory population.")

    ai = NArray(
        label=":math:`a_i`",
        default=numpy.array([0.02, ]),
        domain=Range(lo=0.001, hi=0.1, step=0.001),
        doc="[?]. Decay rate of recovery variable in inhibitory population.")

    bi = NArray(
        label=":math:`b_i`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.05),
        doc="[?]. Slope parameter of u nullcline in inhibitory population.")

    ci = NArray(
        label=":math:`c_i`",
        default=numpy.array([-55.0, ]),
        domain=Range(lo=-100.0, hi=0.0, step=5.0),
        doc="""[mV]. For inhibitory population. Reset value after spike 
                for the original Izhikevich spiking neuronal model.""")

    di = NArray(
        label=":math:`d_i`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.05, hi=8.0, step=0.05),
        doc="""[?]. For inhibitory population. Increment of u after spike
                for the original Izhikevich spiking neuronal model.""")

    fmini = NArray(
        label=":math:`f_{min}_i`",
        default=numpy.array([0.2, ]),
        domain=Range(lo=0.1, hi=1.0, step=0.1),
        doc="""[Hz]. ??Minimum firing rate in inhibitory population.""")

    v_thi = NArray(
        label=":math:`v_{th}_i`",
        default=numpy.array([30.0, ]),
        domain=Range(lo=0.0, hi=100.0, step=5.0),
        doc="""[mV]. For inhibitory population. Membrane potential threshold 
                value for spiking for the original Izhikevich spiking neuronal model.""")

    # Connection weights.
    wee = NArray(
        label=r":math:`w_ee`",
        default=numpy.array([0.1, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence weight (E->E).""")

    wei = NArray(
        label=r":math:`w_ei`",
        default=numpy.array([0.1, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Weight of connection I->E.""")

    wii = NArray(
        label=r":math:`w_ii`",
        default=numpy.array([0.1, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory recurrence weight.""")

    wie = NArray(
        label=r":math:`w_ie`",
        default=numpy.array([0.1, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Weight of connection E->I.""")

    # Shared attributes.
    E_exc = NArray(
        label=":math:`E_{exc}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=100.0, step=5.0),
        doc="""[mV]. Reversal potential in excitatory synapses.""")

    # TODO: check if reversal potential is ok
    E_inh = NArray(
        label=":math:`E_{inh}`",
        default=numpy.array([-90.0, ]),
        domain=Range(lo=-100.0, hi=0.0, step=5.0),
        doc="""[mV]. Reversal potential in inhibitory synapses.""")

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

    g_maxe = NArray(
        label=r":math:`\g_{max}_e`",
        default=numpy.array([0.002, ]),
        domain=Range(lo=0.0, hi=0.1, step=0.001),
        doc="""[nS] Maximum conductance of excitatory synapse.""")

    g_maxi = NArray(
        label=r":math:`\g_{max}_i`",
        default=numpy.array([0.002, ]),
        domain=Range(lo=0.0, hi=0.1, step=0.001),
        doc="""[nS] Maximum conductance of inhibitory synapse.""")

    tau_exc = NArray(
        label=r":math:`\tau_exc`",
        default=numpy.array([10., ]),
        domain=Range(lo=1.0, hi=100.0, step=1.0),
        doc="""[ms]. Time constant of excitatory synapses.""")

    tau_inh = NArray(
        label=r":math:`\tau_inh`",
        default=numpy.array([10., ]),
        domain=Range(lo=1.0, hi=100.0, step=1.0),
        doc="""[ms]. Time constant of inhibitory synapses.""")

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=-100.0, hi=100.0, step=0.1),
        doc="""[pA] Constant background current applied to excitatory population.""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "V_e": numpy.array([None, None]),
                 "V_i": numpy.array([None, None]),
                 "U_e": numpy.array([None, None]),
                 "U_i": numpy.array([None, None]),
                 "R_e": numpy.array([0.0, None]),
                 "R_i": numpy.array([0.0, None]),
                 "I_e": numpy.array([None, None]),
                 "I_i": numpy.array([None, None])
                 },
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. 
            Set None for one-sided boundaries""")
    state_variable_range = Final(
        default={"S_e": numpy.array([0.0, 1.0]),
                 "S_i": numpy.array([0.0, 1.0]),
                 "V_e": numpy.array([-80.0, 30.0]),
                 "V_i": numpy.array([-80.0, 30.0]),
                 "U_e": numpy.array([-15.0, 5.0]),
                 "U_i": numpy.array([-15.0, 5.0]),
                 "R_e": numpy.array([0.0, 1000.0]),
                 "R_i": numpy.array([0.0, 1000.0]),
                 "I_e": numpy.array([-10.0, 10.0]),
                 "I_i": numpy.array([-10.0, 10.0])
                 },

        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i', 'V_e', 'V_i', 'U_e', 'U_i', 'R_e', 'R_i', 'I_e', 'I_i'),
        default=('S_e', 'S_i', 'V_e', 'V_i', 'U_e', 'U_i', 'R_e', 'R_i', 'I_e', 'I_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i', 'V_e', 'V_i', 'U_e', 'U_i', 'R_e', 'R_i', 'I_e', 'I_i']
    non_integrated_variables = ['R_e', 'R_i', 'I_e', 'I_i']
    _nvar = 10
    cvar = numpy.array([0], dtype=numpy.int32)
    _R_e = None
    _R_i = None
    _I_e = None
    _I_i = None
    _J_e = None
    _J_i = None
    _nc_e = None
    _nc_i = None
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
        self.n_nonintvar = self.nvar - self.nintvar
        self._R_e = None
        self._R_i = None
        self._I_e = None  # = I_ext - I_syn
        self._I_i = None
        self._J_e = None  # = I - U
        self._J_i = None
        self._nc_e = None
        self._nc_i = None
        self._ckappa_e = self.ce + self.kappa  # c + kappa
        self._ckappa_i = self.ci + self.kappa
        self._dnck_e = 2 * self.ne2 * self._ckappa_e + self.ne1  # derivative of _nfun at point c + kappa
        self._dnck_i = 2 * self.ni2 * self._ckappa_i + self.ni1
        self._stimulus = 0.0

    def _nfun_e(self, x):
        """This method computes the polynomial term of the dV/dt function"""
        return self.ne2 * x * x + self.ne1 * x + self.ne0 + self._J_e

    def _nfun_i(self, x):
        """This method computes the polynomial term of the dV/dt function"""
        return self.ni2 * x * x + self.ni1 * x + self.ni0 + self._J_i

    def _T(self, sqrtJj, v):
        _T = (10 / sqrtJj) * numpy.arctan(((2 * v) + 125) / (5 * sqrtJj))
        # print("\n_T = \n", _T)
        return _T

    def _f_e(self):
        sqrtJj = numpy.sqrt(4 * (self._J_e + 0j) - 65)
        # print("\nsqrtJj = \n", sqrtJj)
        _f = numpy.maximum(1.0 / (self._T(sqrtJj, self.v_the) - self._T(sqrtJj, self.ce)), self.fmine)
        # print("\n_f = \n", _f)
        return _f

    def _f_i(self):
        sqrtJj = numpy.sqrt(4 * (self._J_i + 0j) - 65)
        # print("\nsqrtJj = \n", sqrtJj)
        _f = numpy.maximum(1.0 / (self._T(sqrtJj, self.v_thi) - self._T(sqrtJj, self.ci)), self.fmini)
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
        S_e = state_variables[0, :]  # synaptic gating dynamics, shape: (brain regions, modes)
        S_i = state_variables[1, :]
        V_e = state_variables[2, :]  # membrane potential, shape: (brain regions, modes)
        V_i = state_variables[3, :]
        U_e = state_variables[4, :]  # recovery variable, shape: (brain regions, modes)
        U_i = state_variables[5, :]

        # coupling is large scale coupling coming from other brain regions, state_variables[cvar], see cvar above:
        # coupling shape: (cvars, brain regions, modes)
        c_0 = coupling[0, :]  # c_0 shape: (brain regions, modes) = sum(w_ij * S_j) for linear additive coupling

        # local coupling for surface simulations, if applicable
        lc_0 = local_coupling * S_e

        # Total current = Iext - Synaptic current
        # = Iext -  (w*S + G*coupling) * g * (V - E_rev)
        self._I_e = self.I_ext - (self.wee * S_e + self.G * (c_0 + lc_0)) * self.g_maxe * (V_e - self.E_exc) \
                    - self.wei * S_i * self.g_maxi * (V_e - self.E_inh)
        self._I_i = - self.wii * S_i * self.g_maxi * (V_i - self.E_inh) \
                    - self.wie * S_e * self.g_maxe * (V_i - self.E_exc)

        # Sigmoidal synaptic function:
        He = 1.0 / (1 + numpy.exp(-self.omega * (V_e - self.ce)))
        Hi = 1.0 / (1 + numpy.exp(-self.omega * (V_i - self.ci)))
        # print("\nH = \n", H)

        # These two will be used later as well for dfun:
        self._J_e = self._I_e - U_e
        self._nc_e = self._nfun_e(self.ce)

        self._J_i = self._I_i - U_i
        self._nc_i = self._nfun_i(self.ci)

        # Rate computation:
        self._R_e = numpy.real(He * numpy.where(self._nc_e > 0.0, self._f_e(), self.fmine))
        self._R_i = numpy.real(Hi * numpy.where(self._nc_i > 0.0, self._f_i(), self.fmini))
        # print("\n_R = \n", self._R)
        # We now update the state_variables vector with the new rate and current:
        state_variables[6, :] = self._R_e.copy()
        state_variables[7, :] = self._R_i.copy()
        state_variables[8, :] = self._I_e.copy()
        state_variables[9, :] = self._I_i.copy()

        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0 * integration_variables[0]] * self.n_nonintvar)

    def _numpy_dfun(self, integration_variables, R_e, R_i):
        r""" TODO: update equations here as well
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}),\\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))},\\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, \gammaH(x_{ek}) \,

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + \lambdaGJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}),\\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))},\\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \,

        """

        S_e = integration_variables[0, :]  # synaptic gating dynamics, shape: (brain regions, modes)
        S_i = integration_variables[1, :]
        V_e = integration_variables[2, :]  # membrane potential, shape: (brain regions, modes)
        V_i = integration_variables[3, :]
        U_e = integration_variables[4, :]  # recovery variable, shape: (brain regions, modes)
        U_i = integration_variables[5, :]

        # Membrane potential dynamics
        ncknc_e = self._nfun_e(self._ckappa_e) - self._nc_e
        eta_e = self._dnck_e / ncknc_e
        xi_e = ncknc_e / numpy.exp(eta_e * self.kappa)
        dV_e = self._nfun_e(V_e) - xi_e * numpy.exp(eta_e * (V_e - self.ce))

        ncknc_i = self._nfun_i(self._ckappa_i) - self._nc_i
        eta_i = self._dnck_i / ncknc_i
        xi_i = ncknc_i / numpy.exp(eta_i * self.kappa)
        dV_i = self._nfun_i(V_i) - xi_i * numpy.exp(eta_i * (V_i - self.ci))

        # Recovery dynamics
        dU_e = self.ae * (self.be * V_e - U_e) + R_e * self.de
        dU_i = self.ai * (self.bi * V_i - U_i) + R_i * self.di

        # Synaptic gating dynamics
        dS_e = R_e - S_e / self.tau_exc
        dS_i = R_i - S_i / self.tau_inh

        return numpy.array([dS_e, dS_i, dV_e, dV_i, dU_e, dU_i])

    def dfun(self, x, c, local_coupling=0.0): # c here is coupling
        if self._R_e is None:
            # Compute intermediate values for this step
            # Form the whole state_variables vector from integration_variables (i.e., x)
            state_variables = self._integration_to_state_variables(x)
            # [S_e, S_i, V_e, V_i, U_e, U_i] + [R_e, R_i, I_e, I_i]
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            R_e = state_variables[6]  # Rates
            R_i = state_variables[7]
            # I_e = state_variables[8]  # Currents
        else:
            # Just set the intermediate values necessary for dfun
            R_e = self._R_e
            R_i = self._R_i
            # I_e = self._I_e
            # if self.use_numba:
            #     deriv = _numba_dfun(x.reshape(x.shape[:-1]).T, R.reshape(x.shape[:-1]).T, I.reshape(x.shape[:-1]).T,
            #                         self.tau_s, self.n0, self.n1, self.n2, self.a, self.b, self.c, self.d,
            #                         self.kappa, self._ckappa, self._dnck)
            #     deriv = deriv.T[..., numpy.newaxis]
            # else:
        deriv = self._numpy_dfun(x, R_e, R_i)  # x = [S, V, U]
        #  Set them to None so that they are recomputed on subsequent steps
        #  for multistep integration schemes such as Runge-Kutta:
        self._R_e = None
        self.R_i = None
        self._I_e = None
        self.I_i = None
        self._J_e = None
        self.J_i = None
        self._nc_e = None
        self.nc_i = None
        return deriv
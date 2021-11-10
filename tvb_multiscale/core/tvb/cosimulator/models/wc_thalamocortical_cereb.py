# -*- coding: utf-8 -*-

import numpy as np
import queue

from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import Model
from tvb.simulator.coupling import Coupling
from tvb.simulator.common import simple_gen_astr


class SafeQueue(queue.Queue):
    empty_output = np.array([0.0])

    def __init__(self, empty_output=np.array([0.0]), *args, **kwargs):
        self.empty_output = empty_output
        super(SafeQueue, self).__init__(*args, **kwargs)

    def put_safe(self, item):
        if self.full():
            self.get()
        super(SafeQueue, self).put(item)

    def get_safe(self):
        if self.full():
            return self.queue[0]
        else:
            return self.empty_output.copy()


class WilsonCowanThalamoCortical(Model):
    r"""
    **References**:

    .. [G_2020] Griffiths, J.D. McIntosh, A.R., Lefebvre J.
       *A Connectome-Based, Corticothalamic Model of State- and Stimulation-Dependent
       Modulation of Rhythmic Neural Activity and Connectivity*,
       Frontiers in Computational Neuroscience, 14:113, 2020,
       10.3389/fncom.2020.575143 .

    """

    # Define traited attributes for this model, these represent possible kwargs.

    is_cortical = NArray(
        label=":math:`is_cortical`",
        dtype=np.bool,
        default=np.array([True]),
        doc="""Boolean flag vector for cortical regions""")

    G = NArray(
        label=":math:`G`",
        default=np.array([5.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Global connectivity scaling factor (g in paper)""")

    w_ee = NArray(
        label=":math:`w_{ee}`",
        default=np.array([0.5]),  # 1.4, 0.5
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory - excitatory gain""")

    w_ei = NArray(
        label=":math:`w_{ei}`",
        default=np.array([1.0]),  # 1.4, 1.0
        domain=Range(lo=0.0, hi=2.0, step=0.02),
        doc="""Excitatory - inhibitory gain""")

    w_ie = NArray(
        label=":math:`w_{ie}`",
        default=np.array([-2.0]),  # -3.0, -2.0
        domain=Range(lo=-4.0, hi=0.0, step=0.04),
        doc="""Inhibitory - excitatory gain.""")

    w_ii = NArray(
        label=":math:`w_{ii}`",
        default=np.array([-0.5]),
        domain=Range(lo=-1.0, hi=0.0, step=0.01),
        doc="""Inhibitory - inhibitory gain.""")

    w_er = NArray(
        label=":math:`w_{er}`",
        default=np.array([0.6]),
        domain=Range(lo=0.0, hi=1.0, step=0.02),
        doc="""Excitatory - reticular gain""")

    w_es = NArray(
        label=":math:`w_{es}`",
        default=np.array([0.6]),
        domain=Range(lo=0.0, hi=1.0, step=0.02),
        doc="""Excitatory - relay gain""")

    w_si = NArray(
        label=":math:`w_{si}`",
        default=np.array([0.2]),
        domain=Range(lo=0.0, hi=0.5, step=0.01),
        doc="""Relay - inhibitory gain.""")

    w_se = NArray(
        label=":math:`w_{se}`",
        default=np.array([1.65]),
        domain=Range(lo=0.0, hi=3.0, step=0.05),
        doc="""Relay - excitatory gain.""")

    w_rs = NArray(
        label=":math:`w_{rs}`",
        default=np.array([-2.0]),
        domain=Range(lo=-4.0, hi=0.0, step=0.4),
        doc="""Reticular - relay gain.""")

    w_sr = NArray(
        label=":math:`w_{sr}`",
        default=np.array([2.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.4),
        doc="""Relay - reticular gain""")

    tau_ct = NArray(
        label=r":math:`\tau_{ct}`",
        default=np.array([20.0]),
        domain=Range(lo=0.1, hi=50.0, step=0.1),
        doc="""Thalamocortical time delay [ms]""")

    tau_tt = NArray(
        label=r":math:`\tau_{tt}`",
        default=np.array([5.0]),
        domain=Range(lo=0.1, hi=10.0, step=0.1),
        doc="""Intrathalamic time delay [ms]""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=np.array([10 / 0.5]),  # 0.9, 0.5
        domain=Range(lo=0.1, hi=10.0, step=0.1),
        doc="""Cortical excitatory population time constant (1/a_e in paper) [ms]""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=np.array([10 / 0.3]),  # 0.9, 0.3
        domain=Range(lo=0.1, hi=10.0, step=0.1),
        doc="""Cortical inhibitory population time constant (1/a_i in paper) [ms]""")

    tau_r = NArray(
        label=r":math:`\tau_r`",
        default=np.array([10 / 0.2]),  # 0.25, 0.2
        domain=Range(lo=0.1, hi=10.0, step=0.1),
        doc="""Thalamic reticular nucleus time constant (1/a_r in paper) [ms]""")

    tau_s = NArray(
        label=r":math:`\tau_s`",
        default=np.array([10 / 0.2]),  # 0.25, 0.2
        domain=Range(lo=0.1, hi=10.0, step=0.1),
        doc="""Thalamic relay nucleus time constant (1/a_s in paper) [ms]""")

    beta = NArray(
        label=":math:`\beta`",
        default=np.array([20.0]),
        domain=Range(lo=0.1, hi=50.0, step=0.1),
        doc="""Activation function gain parameter""")

    sigma = NArray(
        label=":math:`\sigma`",
        default=np.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.1),
        doc="""Activation function threshold parameter""")

    I_e = NArray(
        label=":math:`I_e`",
        default=np.array([-0.35]),
        domain=Range(lo=-1.0, hi=0.0, step=0.05),
        doc="""Cortical excitatory population constant input (i_e in paper)""")

    I_i = NArray(
        label=":math:`I_i`",
        default=np.array([-0.3]),
        domain=Range(lo=-1.0, hi=0.0, step=0.05),
        doc="""Cortical inhibitory population constant input (i_i in paper)""")

    I_r = NArray(
        label=":math:`I_r`",
        default=np.array([-0.8]),
        domain=Range(lo=-1.5, hi=0.0, step=0.05),
        doc="""Thalamic reticular nucleus constant input (i_r in paper)""")

    I_s = NArray(
        label=":math:`I_s`",
        default=np.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        doc="""Thalamic relay nucleus constant input (i_s in paper)""")

    I_o = NArray(
        label=":math:`I_o",
        default=np.array([0.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.05),
        doc="""Static sensory/neuromodulatory drive""")

    M_e = NArray(
        label=":math:`M_e",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Amplitude of sinusoidal stimulus to cortical excitatory population""")

    M_s = NArray(
        label=":math:`M_s",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Amplitude of sinusoidal stimulus to thalamic relay population""")

    f_e = NArray(
        label=":math:`f_e",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Frequency of sinusoidal stimulus to cortical excitatory population""")

    f_s = NArray(
        label=":math:`f_s",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Frequency of sinusoidal stimulus to thalamic relay population""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": np.array([-1.0, 1.0]),
                 "I": np.array([-1.0, 1.0]),
                 "R": np.array([-1.0, 1.0]),
                 "S": np.array([-1.0, 1.0]),
                 "A": np.array([-1.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "R", "S", 'A'),
        default=("E", "I", "R", "S", 'A'),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`
               and :math:`I = 1`.""")

    state_variables = 'E I R S A'.split()
    non_integrated_variables = ['A']
    _nvar = 5
    cvar = np.array([0], dtype=np.int32)
    stvar = np.array([0, 3], dtype=np.int32)

    dt = 0.1
    _n_regions = 1
    _A = None
    _I_so = None
    _et_buffers = None
    _rs_buffers = None
    _sr_buffers = None
    _sc_buffers = None
    _ET_del = None
    _RS_del = None
    _SR_del = None
    _SC_del = None

    _ET_dels = []
    _RS_dels = []
    _SC_dels = []
    _SR_dels = []

    def update_derived_parameters(self):
        self._I_so = self.I_s + self.I_o
        self._n_regions = self.is_cortical.shape[0]
        self.tau_r = self.tau_r.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.tau_s = self.tau_s.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.tau_ct = self.tau_ct.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.tau_tt = self.tau_tt.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        is_subcortical = np.logical_not(self.is_cortical)
        self.tau_r[is_subcortical] = 1.0
        self.tau_s[is_subcortical] = 1.0
        n_delay_steps_ct = (1 + np.round(self.tau_ct / self.dt)).astype('i')
        n_delay_steps_tt = (1 + np.round(self.tau_tt / self.dt)).astype('i')
        dummy = np.zeros((1, self.number_of_modes)).astype('f')
        self._et_buffers = []
        self._sc_buffers = []
        self._rs_buffers = []
        self._sr_buffers = []
        for iR in range(self._n_regions):
            if self.is_cortical[iR]:
                self._et_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_ct[iR]))
                self._rs_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_tt[iR]))
                self._sr_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_tt[iR]))
                self._sc_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_ct[iR]))
        self._ET_del = np.zeros((self._n_regions, self.number_of_modes))
        self._RS_del = np.zeros((self._n_regions, self.number_of_modes))
        self._SR_del = np.zeros((self._n_regions, self.number_of_modes))
        self._SC_del = np.zeros((self._n_regions, self.number_of_modes))
        self.M_e = self.M_e.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.M_s = self.M_s.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.f_e = self.f_e.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self.f_s = self.f_s.reshape((-1, 1)) * np.ones((self._n_regions, 1))
        self._M_s = self.M_s.copy()
        self._M_s[is_subcortical] = 0.0
        self._stim_e = np.zeros((self._n_regions, self.number_of_modes))
        self._stim_s = np.zeros((self._n_regions, self.number_of_modes))
        self._stim_e_inds = self.M_e != 0.0
        self._stim_s_inds = np.logical_and(self._M_s != 0.0, self.is_cortical)
        self._M_e = self.M_e[self._stim_e_inds]
        self._M_s = self._M_s[self._stim_s_inds]
        self._omega_e = 2 * np.pi / 1000 * self.f_e[self._stim_e_inds]  # convert f to angular frequenc omega...
        self._omega_s = 2 * np.pi / 1000 * self.f_s[self._stim_s_inds]  # ...and time for ms to sec

    def sigm_activ(self, x):
        return 1 / (1 + np.exp(-self.beta * (x - self.sigma)))

    def _get_from_buffers(self, buffers):
        output = []
        for buffer in buffers:
            output.append(buffer.get_safe())
        return np.concatenate(output)

    def __ET_del(self):
        return self._get_from_buffers(self._et_buffers)

    def __RS_del(self):
        return self._get_from_buffers(self._rs_buffers)

    def __SR_del(self):
        return self._get_from_buffers(self._sr_buffers)

    def __SC_del(self):
        return self._get_from_buffers(self._sc_buffers)

    def _update_buffers(self, items, buffers):
        for item, buffer in zip(items, buffers):
            buffer.put_safe(item.reshape((-1, 1)))

    def _update_E(self, E):
        self._update_buffers(E, self._et_buffers)

    def _update_R(self, R):
        self._update_buffers(R, self._rs_buffers)

    def _update_S(self, S):
        self._update_buffers(S, self._sr_buffers)
        self._update_buffers(S, self._sc_buffers)

    def __A(self, E, I, S, c_e):
        # local cortico-cortical exc -> exc
        # local cortico-cortical inh -> exc
        # long-range thalamo->cortical relay -> exc
        # zero for subcortical structures
        # long-range cortico-cortical exc -> exc
        return self.w_ee * self.sigm_activ(E) + \
               self.w_ie * self.sigm_activ(I) + \
               np.where(self.is_cortical,
                        self.w_se * self.sigm_activ(self._SC_del),
                        0.0) + \
               self.G * c_e + \
               self.I_e

    def update_state_variables_before_integration(self, state_variables, coupling,
                                                  local_coupling=0.0, stimulus=0.0, time=0.0):
        # This is executed only once for each time step
        # at the beginning of the integration schema computation

        E = state_variables[0]  # exc

        # Update buffers with new input state
        self._update_E(E[self.is_cortical[:, 0]])  # exc
        self._update_R(state_variables[2, self.is_cortical[:, 0]])  # thal reticular
        self._update_S(state_variables[3, self.is_cortical[:, 0]])  # thal relay

        # Get the new delayed state for local thalamocortical and intrathalamic couplings
        self._ET_del[self.is_cortical[:, 0]] = self.__ET_del()  # cortical exc to thalamus delayed coupling
        self._RS_del[self.is_cortical[:, 0]] = self.__RS_del()  # thalamic rtn to thalamic relay delayed coupling
        self._SC_del[self.is_cortical[:, 0]] = self.__SC_del()  # thalamic relay to cortex exc and inh delayed coupling
        self._SR_del[self.is_cortical[:, 0]] = self.__SR_del()  # thalamic relay to thalamic rtn delayed coupling

        self._ET_dels.append(self._ET_del.copy())
        self._RS_dels.append(self._RS_del.copy())
        self._SC_dels.append(self._SC_del.copy())
        self._SR_dels.append(self._SR_del.copy())

        self._stim_e[self._stim_e_inds] = self._M_e * np.sin(self._omega_e * time)
        self._stim_s[self._stim_s_inds] = self._M_s * np.sin(self._omega_s * time)

        # Firing rate/activity of E:
        state_variables[4] = self.__A(state_variables[0, :],  # E, exc
                                      state_variables[1, :],  # I, inh
                                      state_variables[3, :],  # S, thal relay
                                      coupling[0, :])  # c, long-range coupling, Cortical exc
        self._A = state_variables[4]
        return state_variables

    #     def update_state_variables_after_integration(self, state_variables, time=0.0):
    #         return state_variables

    def dfun(self, state_variables, coupling, local_coupling=0.0, time=0.0):
        r"""

        .. math::

        """

        E = state_variables[0, :]  # exc
        I = state_variables[1, :]  # inh
        R = state_variables[2, :]  # thal reticular
        S = state_variables[3, :]  # thal relay

        # long-range coupling
        c_e = coupling[0, :]  # Cortical exc

        derivative = np.empty_like(state_variables)  # no dynamics for A

        derivative[0] = (-E + self._A + self._stim_e) / self.tau_e

        derivative[1] = (-I + \
                         self.w_ii * self.sigm_activ(I) +  # local cortico-cortical inh -> inh
                         self.w_ei * self.sigm_activ(E) +  # local cortico-cortical exc -> inh
                         np.where(self.is_cortical,
                                  self.w_si * self.sigm_activ(self._SC_del),
                                  # long-range thalamo->cortical relay -> inh
                                  0.0) +  # zero for subcortical structures
                         self.I_i) / self.tau_i

        derivative[2] = (-R + \
                         np.where(self.is_cortical,
                                  self.w_er * self.sigm_activ(self._ET_del) +  # cortico-thalamic exc -> reticular
                                  self.w_sr * self.sigm_activ(
                                      self._SR_del) +  # local thalamo-thalamic relay -> reticular
                                  self.I_r,
                                  0.0)) / self.tau_r  # zero for subcortical structures

        derivative[3] = (-S + \
                         np.where(self.is_cortical,
                                  self.w_es * self.sigm_activ(self._ET_del) +  # cortico-thalamic exc -> relay
                                  self.w_rs * self.sigm_activ(
                                      self._RS_del) +  # local thalamo-thalamic reticular -> relay
                                  self._I_so + self._stim_s,
                                  0.0)) / self.tau_s  # zero for subcortical structures

        return derivative


class SigmoidalPre(Coupling):
    r"""
    Provides a sigmoidal coupling function of the form

    .. math::
        c_{min} + (c_{max} - c_{min}) / (1.0 + \exp(-a(x-midpoint)/\sigma))

    NB: using a = numpy.pi / numpy.sqrt(3.0) and the default parameter
        produces something close to the current default for
        Linear (a=0.00390625, b=0) over the linear portion of the sigmoid,
        with saturation at -1 and 1.

    """

    cmin = NArray(
        label=":math:`c_{min}`",
        default=np.array([0.0, ]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="""Minimum of the sigmoid function""", )

    cmax = NArray(
        label=":math:`c_{max}`",
        default=np.array([1.0, ]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="""Maximum of the sigmoid function""", )

    midpoint = NArray(
        label="midpoint",
        default=np.array([0.0, ]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="Midpoint of the linear portion of the sigmoid", )

    a = NArray(
        label=r":math:`a`",
        default=np.array([1.0, ]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="Scaling of sigmoidal", )

    sigma = NArray(
        label=r":math:`\sigma`",
        default=np.array([20.0, ]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="Standard deviation of the sigmoidal", )

    parameter_names = 'cmin cmax midpoint a sigma'.split()
    pre_expr = 'cmin + ((cmax - cmin) / (1.0 + exp(-a *((x_j - midpoint) / sigma))))'
    post_expr = 'gx'

    def __str__(self):
        return simple_gen_astr(self, 'cmin cmax midpoint a sigma')

    def pre(self, x_i, x_j):
        return self.cmin + ((self.cmax - self.cmin) / (1.0 + np.exp(-self.a * ((x_j - self.midpoint) / self.sigma))))

    def post(self, gx):
        return gx

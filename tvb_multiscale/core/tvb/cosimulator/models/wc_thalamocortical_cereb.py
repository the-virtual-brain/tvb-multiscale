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


class GriffithsThalamoCortical(Model):
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
    stvar = np.array([0], dtype=np.int32)

    dt = 0.1
    _n_regions = None
    _n_cortical = None
    _is_subcortical = None

    _A = None

    _I_so = None
    _I_r = None

    _w_er = None
    _w_es = None
    _w_se = None
    _w_si = None
    _w_rs = None
    _w_sr = None

    _tau_ct = None
    _tau_tt = None

    _M_e = None
    _M_s = None
    _omega_e = None
    _omega_s = None
    _stim_e_inds = None
    _stim_s_inds = None
    _stim_e = None
    _stim_s = None

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

    def _configure_params(self):
        self._n_regions = self.is_cortical.shape[0]
        self._is_subcortical = np.logical_not(self.is_cortical)
        self._n_cortical = np.sum(self.is_cortical)
        self._dummy = np.ones((self._n_regions, 1))

    def configure(self):
        super(GriffithsThalamoCortical, self).configure()
        self._configure_params()

    def _assert_size(self, param):
        return param.reshape((-1, 1)) * self._dummy

    def _get_cortical(self, param):
        return self._assert_size(param)[self.is_cortical[:, 0]]

    def _get_subcortical(self, param):
        return self._assert_size(param)[self._is_subcortical[:, 0]]

    def update_derived_parameters(self):
        self._configure_params()

        self._I_so = self._get_cortical(self.I_s + self.I_o)
        self._I_r = self._get_cortical(self.I_r)

        self.tau_r = self._assert_size(self.tau_r)
        self.tau_s = self._assert_size(self.tau_s)
        self.tau_r[self._is_subcortical[:, 0]] = 1.0
        self.tau_s[self._is_subcortical[:, 0]] = 1.0

        self._w_es = self._get_cortical(self.w_es)
        self._w_er = self._get_cortical(self.w_er)
        self._w_se = self._get_cortical(self.w_se)
        self._w_si = self._get_cortical(self.w_si)
        self._w_rs = self._get_cortical(self.w_rs)
        self._w_sr = self._get_cortical(self.w_sr)

        self._tau_ct = self._get_cortical(self.tau_ct)
        self._tau_tt = self._get_cortical(self.tau_tt)
        n_delay_steps_ct = (1 + np.round(self._tau_ct / self.dt)).astype('i')
        n_delay_steps_tt = (1 + np.round(self._tau_tt / self.dt)).astype('i')
        dummy = np.zeros((1, self.number_of_modes)).astype('f')
        self._et_buffers = []
        self._sc_buffers = []
        self._rs_buffers = []
        self._sr_buffers = []
        for iR in range(self._n_cortical):
            self._et_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_ct[iR]))
            self._rs_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_tt[iR]))
            self._sr_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_tt[iR]))
            self._sc_buffers.append(SafeQueue(dummy.copy(), n_delay_steps_ct[iR]))
        self._ET_del = np.zeros((self._n_cortical, self.number_of_modes))
        self._RS_del = np.zeros((self._n_cortical, self.number_of_modes))
        self._SR_del = np.zeros((self._n_cortical, self.number_of_modes))
        self._SC_del = np.zeros((self._n_cortical, self.number_of_modes))

        # Stimulus computation
        self._stim_e_inds = self.M_e != 0.0
        if np.sum(self._stim_e_inds):
            self.M_e = self._assert_size(self.M_e)
            self.f_e = self._assert_size(self.f_e)
            self._stim_e = np.zeros((self._n_regions, 1))
            self._M_e = self.M_e[self._stim_e_inds]
            self._omega_e = 2 * np.pi / 1000 * self.f_e[self._stim_e_inds]  # convert f to angular frequenc omega...
        else:
            self._stim_e = None

        self._stim_s_inds = np.logical_and(self._M_s != 0.0, self.is_cortical)
        if np.sum(self._stim_e_inds):
            self.M_s = self._assert_size(self.M_s)
            self.f_s = self._assert_size(self.f_s)
            self._M_s = self.M_s.copy()
            self._M_s[self._is_subcortical[:, 0]] = 0.0
            self._stim_s = np.zeros((self._n_regions, 1))
            self._stim_s_inds = np.logical_and(self._M_s != 0.0, self.is_cortical)
            self._M_s = self._M_s[self._stim_s_inds]
            self._omega_s = 2 * np.pi / 1000 * self.f_s[self._stim_s_inds]  # ...and time for ms to sec
        else:
            self._stim_s = None

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

    def __E(self, Esigm, Isigm, c):
        """Cortical excitatory population dynamics:
           1. local exc -> exc
           2. local inh -> exc
           3. long-range delayed (sub)cortical (exc) -> exc
           4. local thalamic delayed relay (exc) -> exc, only for cortical regions
        """
        E = self.w_ee * Esigm + \
            self.w_ie * Isigm + \
            self.G * c + \
            self.I_e
        E[self.is_cortical[:, 0]] += self._w_se * self.__SC_del_sigm
        return E

    def __I(self, Esigm, Isigm):
        """Cortical inhibitory population dynamics:
           1. local inh -> inh
           2. local exc -> inh
           3. long-range delayed (sub)cortical -> exc
           4. local delayed thalamic -> exc, only for cortical regions
        """
        I = self.w_ii * Isigm + \
            self.w_ei * Esigm + \
            self.I_i
        I[self.is_cortical[:, 0]] += self._w_si * self.__SC_del_sigm
        return I

    def __R(self):
        """Thalamic relay excitatory population dynamics:
           1. local delayed (sub)cortical -> relay (exc)
           2. local delayed, thalamo-thalamic reticular (inh)-> relay (exc)
        """
        return self._w_er * self.__ET_del_sigm + \
               self._w_sr * self.sigm_activ(self.__SR_del()) + \
               self._I_r

    def __S(self):
        """Thalamic relay excitatory population dynamics:
           1. local delayed (sub)cortical -> relay (exc)
           2. local delayed, thalamo-thalamic reticular (inh)-> relay (exc)
        """
        return self._w_es * self.__ET_del_sigm + \
               self._w_rs * self.sigm_activ(self.__RS_del()) + \
               self._I_so

    def update_state_variables_before_integration(self, state_variables, coupling,
                                                  local_coupling=0.0, stimulus=0.0, time=0.0):
        # This is executed only once for each time step
        # at the beginning of the integration schema computation

        # Update buffers with new input state
        self._update_E(state_variables[0, self.is_cortical[:, 0]])  # Cortical exc
        self._update_R(state_variables[2, self.is_cortical[:, 0]])  # thal reticular
        self._update_S(state_variables[3, self.is_cortical[:, 0]])  # thal relay

        # Get the new delayed state for local thalamocortical and intrathalamic couplings
        self._ET_del = self.__ET_del()  # cortical exc to thalamus delayed coupling
        self._RS_del = self.__RS_del()  # thalamic rtn to thalamic relay delayed coupling
        self._SC_del = self.__SC_del()  # thalamic relay to cortex exc and inh delayed coupling
        self._SR_del = self.__SR_del()  # thalamic relay to thalamic rtn delayed coupling
        self.__ET_del_sigm = self.sigm_activ(self._ET_del)
        self.__SC_del_sigm = self.sigm_activ(self._SC_del)

        # These store delays only for testing:
        self._ET_dels.append(self._ET_del.copy())
        self._RS_dels.append(self._RS_del.copy())
        self._SC_dels.append(self._SC_del.copy())
        self._SR_dels.append(self._SR_del.copy())

        # Firing rate/activity of E:
        state_variables[4] = self.__E(self.sigm_activ(state_variables[0, :]),  # E, exc
                                      self.sigm_activ(state_variables[1, :]),  # I, inh
                                      coupling[0, :])  # c, long-range coupling, Cortical exc

        # Store this temporarily to avoid double computation:
        self._A = state_variables[4]

        # Compute stimulus, if any, for this time point:
        if self._stim_e is not None:
            self._stim_e = self._M_e * np.sin(self._omega_e * time)
        if self._stim_s is not None:
            self._stim_s = self._M_s * np.sin(self._omega_s * time)

        return state_variables

    #     def update_state_variables_after_integration(self, state_variables, time=0.0):
    #         return state_variables

    def dfun(self, state_variables, coupling, local_coupling=0.0, time=0.0):
        r"""

        .. math::

        """

        # E = state_variables[0, :]  # exc
        # I = state_variables[1, :]  # inh
        # R = state_variables[2, :]  # thal reticular
        # S = state_variables[3, :]  # thal relay

        # # long-range coupling
        # c_e = coupling[0, :]  # Cortical exc

        derivative = np.empty_like(state_variables)  # no dynamics for A

        Esigm = self.sigm_activ(state_variables[0, :])
        Isigm = self.sigm_activ(state_variables[0, :])

        if self._A is None:
            self._A = self.__A(Esigm,           # E, exc
                               Isigm,           # I, inh
                               coupling[0, :])  # c, long-range coupling, Cortical exc

        if self._stim_e is not None:
            self._A[self._stim_e_inds] = self._A[self._stim_e_inds] + self._stim_e

        derivative[0] = (-state_variables[0, :] + self._A) / self.tau_e
        # Set it back to None after using it:
        self._A = None

        derivative[1] = (-state_variables[1, :] +
                         + self.__I(Esigm,           # E, exc
                                    Isigm            # I, inh
                                    ))/self.tau_i

        derivative[2] = -state_variables[2, :]
        derivative[2, self.is_cortical[:, 0]] += self.__R()
        derivative[2] /= self.tau_r

        derivative[3] = -state_variables[3, :]
        derivative[3, self.is_cortical[:, 0]] += self.__S()
        if self._stim_s is not None:
            derivative[3, self._stim_s_inds] = derivative[3, self._stim_s_inds] + self._stim_s
        derivative[3] /= self.tau_s

        return derivative


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

    is_thalamic = NArray(
        label=":math:`is_thalamic`",
        dtype=np.bool,
        default=np.array([False]),
        doc="""Boolean flag vector for thalamic regions""")

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

    A_st = NArray(
        label=":math:`A_st",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Amplitude of sinusoidal stimulus to excitatory population""")

    f_st = NArray(
        label=":math:`f_st",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Frequency of sinusoidal stimulus to excitatory population""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": np.array([-1.0, 1.0]),
                 "I": np.array([-1.0, 1.0]),
                 "A": np.array([-1.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", 'A'),
        default=("E", "I", 'A'),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`
               and :math:`I = 1`.""")

    state_variables = 'E I A'.split()
    non_integrated_variables = ['A']
    _nvar = 3
    cvar = np.array([0], dtype=np.int32)
    stvar = np.array([0], dtype=np.int32)

    dt = 0.1
    _n_regions = 1
    _n_thalamic = 0
    _not_thalamic = None

    _tau_e = None
    _tau_i = None
    _tau_r = None

    _G_e = None
    _G_th = None

    _w_ee = None
    _w_ie = None
    _w_ei = None
    _w_ii = None
    _w_es = None
    _w_er = None
    _w_se = None
    _w_si = None
    _w_rs = None
    _w_sr = None

    _I_e = None
    _I_i = None
    _I_so = None
    _I_r = None

    _E = None
    _stim = None

    _rs_buffers = []
    _sr_buffers = []
    _RS_del = np.zeros((0, 0))
    _SR_del = np.zeros((0, 0))
    _RS_dels = []
    _SR_dels = []

    def _configure_params(self):
        self._n_regions = self.is_thalamic.shape[0]
        self._n_thalamic = np.sum(self.is_thalamic)
        self._not_thalamic = np.logical_not(self.is_thalamic)
        self._dummy = np.ones((self._n_regions, 1))

    def configure(self):
        super(WilsonCowanThalamoCortical, self).configure()
        self._configure_params()

    def _assert_size(self, param):
        return param.reshape((-1, 1)) * self._dummy

    def _get_thalamic(self, param):
        return self._assert_size(param)[self.is_thalamic[:, 0]]

    def _get_nonthalamic(self, param):
        return self._assert_size(param)[self._not_thalamic[:, 0]]

    def update_derived_parameters(self):

        self._configure_params()

        self._I_e = self._get_nonthalamic(self.I_e)
        self._I_i = self._get_nonthalamic(self.I_i)
        self._I_so = self._get_thalamic(self.I_s + self.I_o)
        self._I_r = self._get_thalamic(self.I_r)

        self._tau_e = np.where(self.is_thalamic, self.tau_s, self.tau_e)
        self._tau_i = self._get_nonthalamic(self.tau_i)
        self._tau_r = self._get_thalamic(self.tau_r)

        self._G_e = self._get_nonthalamic(self.G)
        self._G_th = self._get_thalamic(self.G)

        self._w_ee = self._get_nonthalamic(self.w_ee)
        self._w_ei = self._get_nonthalamic(self.w_ei)
        self._w_ie = self._get_nonthalamic(self.w_ie)
        self._w_ii = self._get_nonthalamic(self.w_ii)
        self._w_se = self._get_nonthalamic(self.w_se)
        self._w_si = self._get_nonthalamic(self.w_si)

        self._w_es = self._get_thalamic(self.w_es)
        self._w_er = self._get_thalamic(self.w_er)
        self._w_rs = self._get_thalamic(self.w_rs)
        self._w_sr = self._get_thalamic(self.w_sr)

        # Prepare buffers for thalamic relay <-> relay connections' delays:
        self._rs_buffers = []
        self._sr_buffers = []
        self._RS_del = np.zeros((self._n_thalamic, self.number_of_modes))
        self._SR_del = np.zeros((self._n_thalamic, self.number_of_modes))
        if self._n_thalamic:
            tau_tt = self._get_thalamic(self.tau_tt)
            n_delay_steps_tt = (1 + np.round(tau_tt / self.dt)).astype('i')
            dummy = np.zeros((1, self.number_of_modes)).astype('f')
            for n_delay_step_tt in n_delay_steps_tt:
                self._rs_buffers.append(SafeQueue(dummy.copy(), n_delay_step_tt))
                self._sr_buffers.append(SafeQueue(dummy.copy(), n_delay_step_tt))

        # Prepare stimulus only for those regions that have some
        self.A_st = self._assert_size(self.A_st)
        self._stim_inds = self.A_st != 0.0
        if np.sum(self._stim_inds):
            self._A_st = self.A_st[self._stim_inds]
            # convert f to angular frequency omega:
            self._omega_st = 2 * np.pi / 1000 * self._assert_size(self.f_st)[self._stim_inds]
            self._stim = 1
        else:
            self._stim = None

    def sigm_activ(self, x):
        return 1 / (1 + np.exp(-self.beta * (x - self.sigma)))

    def _get_from_buffers(self, buffers):
        output = []
        for buffer in buffers:
            output.append(buffer.get_safe())
            return np.concatenate(output)

    def __RS_del(self):
        return self._get_from_buffers(self._rs_buffers)

    def __SR_del(self):
        return self._get_from_buffers(self._sr_buffers)

    def _update_buffers(self, items, buffers):
        for item, buffer in zip(items, buffers):
            buffer.put_safe(item.reshape((-1, 1)))

    def _update_R(self, R):
        self._update_buffers(R, self._rs_buffers)

    def _update_S(self, S):
        self._update_buffers(S, self._sr_buffers)

    def __E(self, Esigm, Isigm, c_cx, c_sb, c_th):
        """Cortical excitatory population dynamics:
           1. local exc -> exc
           2. local inh -> exc
           3. long-range delayed (sub)cortical (exc) -> exc
           4. long-range delayed thalamic relay (exc) -> exc
        """
        return self._w_ee * Esigm + \
               self._w_ie * Isigm + \
               self._G_e * (c_cx + c_sb) + \
               self._w_se * c_th + \
               self._I_e

    def __I(self, Esigm, Isigm, c_th):
        """Cortical inhibitory population dynamics:
           1. local inh -> inh
           2. local exc -> inh
           3. long-range delayed thalamic -> exc
        """
        return self._w_ii * Isigm + \
               self._w_ei * Esigm + \
               self._w_si * c_th + \
               self._I_i

    def __S(self, c_cx, c_sb):
        """Thalamic relay excitatory population dynamics:
           1. local delayed, thalamo-thalamic reticular (inh)-> relay (exc)
           2. long-range delayed (sub)cortical -> relay (exc)
        """
        return self._w_rs * self.sigm_activ(self.__RS_del()) + \
               self._w_es * c_cx + self._G_th * c_sb + \
               self._I_so

    def __R(self, c_cx):
        """Thalamic relay excitatory population dynamics:
           1. local delayed, thalamo-thalamic reticular (inh)-> relay (exc)
           2. long-range delayed (sub)cortical -> relay (exc)
        """
        return self._w_sr * self.sigm_activ(self.__SR_del()) + \
               self._w_er * c_cx + \
               self._I_r

    def update_state_variables_before_integration(self, state_variables, coupling,
                                                  local_coupling=0.0, stimulus=0.0, time=0.0):
        # This is executed only once for each time step
        # at the beginning of the integration schema computation

        # Firing rate/activity of
        # ...cortical E:
        state_variables[2, self._not_thalamic[:, 0]] = \
            self.__E(self.sigm_activ(state_variables[0, self._not_thalamic[:, 0]]),  # E, exc
                     self.sigm_activ(state_variables[1, self._not_thalamic[:, 0]]),  # I, inh
                     coupling[0, self._not_thalamic[:, 0]],  # c_cx, long-range coupling, Cortical exc
                     coupling[1, self._not_thalamic[:, 0]],  # c_sb, long-range coupling, Subcortical exc
                     coupling[2, self._not_thalamic[:, 0]])  # c_th, long-range coupling, Thalamic relay (exc)

        if self._n_thalamic:

            # First, update buffers with new input state
            self._update_S(state_variables[0, self.is_thalamic[:, 0]])  # thal relay
            self._update_R(state_variables[1, self.is_thalamic[:, 0]])  # thal reticular

            # Get the new delayed state for local thalamocortical and intrathalamic couplings
            self._RS_del = self.__RS_del()  # thalamic rtn to thalamic relay delayed coupling
            self._SR_del = self.__SR_del()  # thalamic relay to thalamic rtn delayed coupling

            # These store delays only for testing:
            self._RS_dels.append(self._RS_del.copy())
            self._SR_dels.append(self._SR_del.copy())

            # ...thalamic relay S (Exc):
            state_variables[2, self.is_thalamic[:, 0]] = \
                self.__S(coupling[0, self.is_thalamic[:, 0]],  # c_cx, long-range coupling, Cortical exc
                         coupling[1, self.is_thalamic[:, 0]])  # c_sb, long-range coupling, Subcortical exc

        # Store this temporarily to avoid double computation:
        self._E = state_variables[2, :].copy()

        # Add the stimulus:
        if self._stim is not None:
            self._stim = self._A_st * np.sin(self._omega_st * time)

        return state_variables

    #     def update_state_variables_after_integration(self, state_variables, time=0.0):
    #         return state_variables

    def dfun(self, state_variables, coupling, local_coupling=0.0, time=0.0):
        r"""

        .. math::

        """

        E = state_variables[0, :]  # (sub)cortical exc or thalamic relay (exc)
        I = state_variables[1, :]  # (sub)cortical inh or thalamic reticular (inh)

        # long-range coupling
        c_cx = coupling[0, :]  # cortical exc
        c_sb = coupling[1, :]  # subcortical exc
        c_th = coupling[2, :]  # thalamic relay (exc)

        derivative = np.empty_like(state_variables)  # no dynamics for A

        sigmE = self.sigm_activ(E[self._not_thalamic[:, 0]])
        sigmI = self.sigm_activ(I[self._not_thalamic[:, 0]])

        if self._E is None:
            self._E = derivative[0].copy()
            self._E[self._not_thalamic[:, 0]] = \
                self.__E(sigmE,                             # E, exc
                         sigmI,                             # I, inh
                         c_cx[self._not_thalamic[:, 0]],    # c_cx, long-range coupling, Cortical exc
                         c_sb[self._not_thalamic[:, 0]],    # c_sb, long-range coupling, Subcortical exc
                         c_th[self._not_thalamic[:, 0]])    # c_th, long-range coupling, Thalamic relay (exc)
            if self._n_thalamic:
                # ...thalamic relay S (Exc):
                self._E[self.is_thalamic[:, 0]] = \
                    self.__S(
                        c_cx[self.is_thalamic[:, 0]],        # c_cx, long-range coupling, Cortical exc
                        c_sb[self.is_thalamic[:, 0]])        # c_sb, long-range coupling, Subcortical exc

        derivative[1, self._not_thalamic[:, 0]] = (
            - I[self._not_thalamic[:, 0]] +
            + self.__I(sigmE,                                # E, exc
                       sigmI,                                # I, inh
                       c_th[self._not_thalamic[:, 0]]        # c_th, long-range coupling, Thalamic relay (exc)
                       )
            ) / self._tau_i
        if self._n_thalamic:
            derivative[1, self.is_thalamic[:, 0]] = (
                - I[self.is_thalamic[:, 0]]
                + self.__R(c_cx[self.is_thalamic[:, 0]])     # c_cx, long-range coupling, Cortical exc
                ) / self._tau_r

        if self._stim is not None:
            self._E[self._stim_inds] = self._E[self._stim_inds] + self._stim

        derivative[0, :] = (- E + self._E) / self._tau_e

        # Set it back to None after using it:
        self._E = None

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

    def configure(self):
        super(SigmoidalPre, self).configure()
        self.c_max_min = self.cmax - self.cmin

    def pre(self, x_i, x_j):
        return self.cmin + (self.c_max_min / (1.0 + np.exp(-self.a * ((x_j - self.midpoint) / self.sigma))))

    def post(self, gx):
        return gx


class SigmoidalPreThalamoCortical(SigmoidalPre):

    is_thalamic = NArray(
        label=":math:`is_thalamic`",
        dtype=np.bool,
        default=np.array([False]),
        doc="""Boolean flag vector for specific thalamic nuclei regions""")

    is_subcortical = NArray(
        label=":math:`is_subcortical`",
        dtype=np.bool,
        default=np.array([False]),
        doc="""Boolean flag vector for subcortical regions, including specific thalamic nuclei ones""")

    _cx_inds = None
    _sb_inds = None
    _th_inds = None
    _n_cvars = None
    _n_ctypes = 1
    _pre = None
    _sb_cvar = None
    _sb_cvars = None
    _th_cvar = None
    _th_cvars = None
    _cx_cvars = None

    def configure(self):
        super(SigmoidalPreThalamoCortical, self).configure()
        self._cx_inds = np.where(np.logical_not(self.is_subcortical))[0]
        self._sb_inds = np.where(np.logical_and(self.is_subcortical,
                                                np.logical_not(self.is_thalamic)))[0]
        self._th_inds = np.where(self.is_thalamic)[0]
        self._sb_cvar = 0
        self._th_cvar = 0
        if len(self._sb_inds):
            self._sb_cvar = 1
            self._n_ctypes += 1
        if np.any(self.is_thalamic):
            self._n_ctypes += 1
            self._th_cvar = self._sb_cvar + 1
        self._n_cvars = None
        self._pre = None
        for p in ["cmin", "c_max_min", "midpoint", "a", "sigma"]:
            setattr(self, "_" + p + "_cx", None)
            setattr(self, "_" + p + "_sb", None)
            setattr(self, "_" + p + "_th", None)

    def _shape_parameters(self):
        if self._pre is not None:
            for p in ["cmin", "c_max_min", "midpoint", "a", "sigma"]:
                temp = getattr(self, p)
                while temp.ndim < 4:
                    temp = temp[..., np.newaxis]
                temp = temp * np.ones(self._pre.shape)
                setattr(self, "_" + p + "_cx",
                        temp[:, self._cx_cvars[:, None], self._cx_inds[None, :], :])
                if self._sb_cvar:
                    setattr(self, "_" + p + "_sb",
                            temp[:, self._sb_cvars[:, None], self._sb_inds[None, :], :])
                if self._th_cvar:
                    setattr(self, "_" + p + "_th",
                            temp[:, self._th_cvars[:, None], self._th_inds[None, :], :])

    def pre(self, x_i, x_j):
        # Modification to double the coupling variables,
        # in order to separate thalamic from non-thalamic regins' inputs x_j:
        if self._pre is None:
            # It runs only for the first time
            pre_shape = list(x_j.shape)
            self._n_cvars = pre_shape[1]
            pre_shape[1] *= self._n_ctypes
            self._cx_cvars = np.arange(self._n_cvars)
            if self._sb_cvar:
                self._sb_cvars = np.arange(self._n_cvars, 2 * self._n_cvars)
            if self._th_cvar:
                self._th_cvars = np.arange(self._th_cvar * self._n_cvars, (self._th_cvar + 1) * self._n_cvars)
            self._pre = np.zeros(tuple(pre_shape)).astype(x_j.dtype)
            self._shape_parameters()
        # The first coupling variables uses the non-thalamic regions inputs
        self._pre[:, self._cx_cvars[:, None], self._cx_inds[None, :], :] = \
            self._cmin_cx + (self._c_max_min_cx /
                             (1.0 + np.exp(-self._a_cx * ((x_j[:, :, self._cx_inds, :]
                                                          - self._midpoint_cx) / self._sigma_cx))))
        # The second coupling variables uses the subcortical, but non-specific thalamic regions, inputs
        if self._sb_cvar:
            self._pre[:, self._sb_cvars[:, None], self._sb_inds[None, :], :] = \
                self._cmin_sb + (self._c_max_min_sb /
                                 (1.0 + np.exp(-self._a_sb * ((x_j[:, :, self._sb_inds, :]
                                                              - self._midpoint_sb) / self._sigma_sb))))
        # The third coupling variables uses the specific thalamic regions inputs
        if self._th_cvar:
            self._pre[:, self._th_cvars[:, None], self._th_inds[None, :], :] = \
                self._cmin_th + (self._c_max_min_th /
                                 (1.0 + np.exp(-self._a_th * ((x_j[:, :, self._th_inds, :]
                                                              - self._midpoint_th) / self._sigma_th))))
        return self._pre.copy()

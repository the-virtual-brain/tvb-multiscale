
import math
import numpy as np
from tvb.simulator.models.base import ModelNumbaDfun, Model
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final


class MaithEtAlCortexExcInhGC(Model):

    n_0 = NArray(
        label=r":math:`n_0`",
        default=np.array([140.0]),
        domain=Range(lo=0.0, hi=200.0, step=0.01),
        doc="""Scaling factor of zero order of membrane potential."""
    )
    n_1 = NArray(
        label=r":math:`n_1`",
        default=np.array([5.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Scaling factor of first order of membrane potential."""
    )
    n_2 = NArray(
        label=r":math:`n_2`",
        default=np.array([0.04]),
        domain=Range(lo=0.0, hi=0.1, step=0.01),
        doc="""Scaling factor of second order of membrane potential."""
    )
    C = NArray(
        label=r":math:`C`",
        default=np.array([1.0]),
        domain=Range(lo=0.01, hi=2.0, step=0.01),
        doc="""Membrane capacitance."""
    )
    I_e = NArray(
        label=r":math:`I_e`",
        default=np.array([50.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.5),
        doc="""Excitatory external input."""
    )
    I_i = NArray(
        label=r":math:`I_i`",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.5),
        doc="""Inhibitory external input."""
    )
    a = NArray(
        label=r":math:`a`",
        default=np.array([0.02]),
        domain=Range(lo=0.0, hi=0.04, step=0.001),
        doc="""Rate of adaptation."""
    )
    b = NArray(
        label=r":math:`b`",
        default=np.array([0.2]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Sensitivity of adaptation to subthreshold fluctuations of membrane potential."""
    )
    c = NArray(
        label=r":math:`c`",
        default=np.array([-72.0]),
        domain=Range(lo=-100.0, hi=0.0, step=1.0),
        doc="""After-spike reset memprane potential ."""
    )
    d = NArray(
        label=r":math:`d`",
        default=np.array([6.0]),
        domain=Range(lo=0.0, hi=500.0, step=0.01),
        doc="""After-spike adaptation increment."""
    )
    E_AMPA = NArray(
        label=r":math:`E_AMPA`",
        default=np.array([0.0]),
        domain=Range(lo=-50.0, hi=0.0, step=1.0),
        doc="""AMPA reversal potential."""
    )
    E_GABA = NArray(
        label=r":math:`E_GABA`",
        default=np.array([-90.0]),
        domain=Range(lo=-100.0, hi=0.0, step=1.0),
        doc="""GABA reversal potential."""
    )
    τ_AMPA = NArray(
        label=r":math:`τ_AMPA`",
        default=np.array([10.0]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="""AMPA synaptic time constant."""
    )
    τ_GABA = NArray(
        label=r":math:`τ_GABA`",
        default=np.array([10.0]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="""GABA synaptic time constant."""
    )
    G_cAMPA = NArray(
        label=r":math:`G_cAMPA`",
        default=np.array([0.0]),
        domain=Range(lo=0.01, hi=10.0, step=0.01),
        doc="""AMPA global coupling strength to excitatory population."""
    )
    cAMPA_e = NArray(
        label=r":math:`cAMPA_e`",
        default=np.array([0.0]),
        domain=Range(lo=0.01, hi=10.0, step=0.01),
        doc="""AMPA coupling strength to excitatory population."""
    )
    cGABA_e = NArray(
        label=r":math:`cGABA_e`",
        default=np.array([0.166]),
        domain=Range(lo=0.01, hi=10.0, step=0.01),
        doc="""GABA coupling strength to excitatory population."""
    )
    cAMPA_i = NArray(
        label=r":math:`cAMPA_i`",
        default=np.array([1.0]),
        domain=Range(lo=0.01, hi=1.0, step=0.01),
        doc="""AMPA coupling strength to inhibitory population."""
    )
    cGABA_i = NArray(
        label=r":math:`cGABA_i`",
        default=np.array([0.2]),
        domain=Range(lo=0.01, hi=10.0, step=0.01),
        doc="""GABA coupling strength to inhibitory population."""
    )
    Δ = NArray(
        label=r":math:`Δ`",
        default=np.array([1.0]),
        domain=Range(lo=0.001, hi=1.0, step=0.001),
        doc="""Width of Lorentzian distribution of external input."""
    )
    S_jump = NArray(
        label=r":math:`S_jump`",
        default=np.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""After-spike conductance increment."""
    )

    _nvar = 8
    state_variables = ['r_e', 'r_i', 'v_e', 'v_i', 'w_e', 'w_i', 'g_ampa', 'g_gaba']

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "r_e": np.array([0.0, 0.0]),
			"r_i": np.array([0.0, 0.0]),
			"v_e": np.array([0.0, 0.0]),
			"v_i": np.array([0.0, 0.0]),
			"w_e": np.array([0.0, 0.0]),
			"w_i": np.array([0.0, 0.0]),
			"g_ampa": np.array([0.0, 0.0]),
			"g_gaba": np.array([0.0, 0.0])
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup."""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r_e": np.array([0.0, 1e9]),
			"r_i": np.array([0.0, 1e9]),
			"v_e": np.array([-1e9, 1e9]),
			"v_i": np.array([-1e9, 1e9]),
			"w_e": np.array([-1e9, 1e9]),
			"w_i": np.array([-1e9, 1e9]),
			"g_ampa": np.array([0.0, 1e9]),
			"g_gaba": np.array([0.0, 1e9])
        },
        doc="""State variable boundaries for phase plane setup."""
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('r_e', 'r_i', 'v_e', 'v_i', 'w_e', 'w_i', 'g_ampa', 'g_gaba',
                 'V_e', 'V_i', 'W_e', 'W_i', 'G_AMPA', 'G_GABA', 'I_AMPA_e', 'I_AMPA_i', 'I_GABA_e', 'I_GABA_i'),
        default=('r_e', 'r_i', 'v_e', 'v_i', 'w_e', 'w_i', 'g_ampa', 'g_gaba',
                 'V_e', 'V_i', 'W_e', 'W_i', 'G_AMPA', 'G_GABA', 'I_AMPA_e', 'I_AMPA_i', 'I_GABA_e', 'I_GABA_i'),
        doc="""Default state variables to be monitored."""
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=['n_0', 'n_1', 'n_2', 'C', 'I_e', 'I_i', 'a', 'b', 'c', 'd',
                 'E_AMPA', 'E_GABA', 'τ_AMPA', 'τ_GABA',
                 'G_cAMPA', 'cAMPA_e', 'cGABA_e', 'cAMPA_i', 'cGABA_i',
                 'Δ', 'S_jump'],
    )

    local_parameter_names = []

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "r_e": "(Δ_hat / (τ*π) + (2*v_e - α - cAMPA_e*g_ampa - G_cAMPA * c_g_ampa - cGABA_e*g_gaba)*r_e)/τ",
            "r_i": "(Δ_hat / (τ*π) + (2*v_i - α - cAMPA_i*g_ampa - cGABA_i*g_gaba)*r_i)/τ",
            "v_e": "(v_e**2 - α*v_e - (τ*π*r_e)**2 - w_e + i_e"
                   " + (cAMPA_e*g_ampa + G_cAMPA*c_g_ampa)*(e_ampa-v_e) + cGABA_e*g_gaba*(e_gaba-v_e))/τ",
            "v_i": "(v_i**2 - α*v_i - (τ*π*r_i)**2 - w_i + i_i"
                   " + cAMPA_i*g_ampa*(e_ampa-v_i) + cGABA_i*g_gaba*(e_gaba-v_i))/τ",
            "w_e": "a*(b_hat*v_e - w_e) + w_jump*r_e - a*b_hat",
            "w_i": "a*(b_hat*v_i - w_i) + w_jump*r_i - a*b_hat",
            "g_ampa": "-g_ampa/τ_AMPA + s_jump * r_e",
            "g_gaba": "-g_gaba/τ_GABA + s_jump * r_i",
        },
    )

    cvar = np.array([6], dtype=np.int32)

    coupling_terms = Final(
        label="Coupling terms",
        default=['c_g_ampa']
    )

    _R = None
    _stimulus = 0.0
    use_numba = False

    def update_derived_parameters(self):

        # Mathematical functions
        exp = np.exp
        sqrt = np.sqrt

        n_0 = self.n_0
        n_1 = self.n_1
        n_2 = self.n_2
        C = self.C
        I_e = self.I_e
        I_i = self.I_i
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        E_AMPA = self.E_AMPA
        E_GABA = self.E_GABA
        τ_AMPA = self.τ_AMPA
        τ_GABA = self.τ_GABA
        G_cAMPA = self.G_cAMPA
        cAMPA_e = self.cAMPA_e
        cGABA_e = self.cGABA_e
        cAMPA_i = self.cAMPA_i
        cGABA_i = self.cGABA_i
        Δ = self.Δ
        S_jump = self.S_jump

        self.k_1 = k_1 = C * n_2
        self.V_R = V_R = 1/(2*n_2) * (-n_1 - sqrt(n_1**2 - 4*n_0*n_2))
        self.V_T = V_T = 1/(2*n_2) * (-n_1 + sqrt(n_1**2 - 4*n_0*n_2))
        self.α = α = 1 + V_T/abs(V_R)
        self.e_ampa = e_ampa = 1 + E_AMPA/abs(V_R)
        self.e_gaba = e_gaba = 1 + E_GABA/abs(V_R)
        self.g_sc = g_sc = k_1 * abs(V_R)
        self.b_hat = b_hat = b / g_sc
        self.τ = τ = C / g_sc
        self.s_jump = s_jump = S_jump * τ
        self.i_sc = i_sc = k_1 * V_R**2
        self.w_jump = w_jump = d / i_sc
        self.Δ_hat = Δ_hat = C * Δ / i_sc
        self.i_e = i_e = C * I_e / i_sc
        self.i_i = i_i = C * I_i / i_sc

    def _build_observer(self):
        template = ("def observe(state):\n"
                    "    {svars} = state\n"
                    "    V_e = (v_e - 1) * abs(V_R)\n"
                    "    V_i = (v_i - 1) * abs(V_R)\n"
                    "    W_e = w_e * i_sc\n"
                    "    W_i = w_i * i_sc\n"
                    "    G_AMPA = g_ampa * g_sc\n"
                    "    G_GABA = g_gaba * g_sc\n"
                    "    I_AMPA_e = cAMPA_e * G_AMPA * (E_AMPA - V_e)\n"
                    "    I_AMPA_i = cAMPA_i * G_AMPA * (E_AMPA - V_i)\n"
                    "    I_GABA_e = cGABA_e * G_GABA * (E_GABA - V_e)\n"
                    "    I_GABA_i = cGABA_i * G_GABA * (E_GABA - V_i)\n"
                    "    return numpy.array([{voi_names}])")
        svars = ','.join(self.state_variables)
        if len(self.state_variables) == 1:
            svars += ','
        code = template.format(
            svars=svars,
            voi_names=','.join(self.variables_of_interest)
        )
        namespace = {'numpy': np}
        namespace.update(self.__dict__)
        self.log.debug('building observer with code:\n%s', code)
        exec(code, namespace)
        self.observe = namespace['observe']
        self.observe.code = code

    def dfun(self, state_variables, coupling, local_coupling=0):

        # Mathematical functions
        exp = np.exp
        sqrt = np.sqrt

        # shape (n_sv, n_modes)
        r_e = state_variables[0, :]
        r_i = state_variables[1, :]
        v_e = state_variables[2, :]
        v_i = state_variables[3, :]
        w_e = state_variables[4, :]
        w_i = state_variables[5, :]
        g_ampa = state_variables[6, :]
        g_gaba = state_variables[7, :]

        # Parameters
        pi = π = np.pi

        n_0 = self.n_0
        n_1 = self.n_1
        n_2 = self.n_2
        C = self.C
        I_e = self.I_e
        I_i = self.I_i
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        E_AMPA = self.E_AMPA
        E_GABA = self.E_GABA
        τ_AMPA = self.τ_AMPA
        τ_GABA = self.τ_GABA
        G_cAMPA = self.G_cAMPA
        cAMPA_e = self.cAMPA_e
        cGABA_e = self.cGABA_e
        cAMPA_i = self.cAMPA_i
        cGABA_i = self.cGABA_i
        Δ = self.Δ
        S_jump = self.S_jump

        k_1 = self.k_1
        V_R = self.V_R
        V_T = self.V_T
        α = self.α
        e_ampa = self.e_ampa
        e_gaba = self.e_gaba
        g_sc = self.g_sc
        b_hat = self.b_hat
        τ = self.τ
        s_jump = self.s_jump
        i_sc = self.i_sc
        w_jump = self.w_jump
        Δ_hat = self.Δ_hat
        i_e = self.i_e
        i_i = self.i_i

        # Coupling Terms
        c_g_ampa = coupling[0, :]

        # Derived Variables

        # Time Derivatives
        derivative = np.empty_like(state_variables)

        derivative[0] = (Δ_hat / (τ*π) + (2*v_e - α - cAMPA_e*g_ampa - G_cAMPA * c_g_ampa - cGABA_e*g_gaba)*r_e)/τ
        derivative[1] = (Δ_hat / (τ*π) + (2*v_i - α - cAMPA_i*g_ampa - cGABA_i*g_gaba)*r_i)/τ
        derivative[2] = (v_e**2 - α*v_e - (τ*π*r_e)**2 - w_e + i_e
                         + (cAMPA_e*g_ampa + G_cAMPA*c_g_ampa)*(e_ampa-v_e) + cGABA_e*g_gaba*(e_gaba-v_e))/τ
        derivative[3] = (v_i**2 - α*v_i - (τ*π*r_i)**2 - w_i + i_i
                         + cAMPA_i*g_ampa*(e_ampa-v_i) + cGABA_i*g_gaba*(e_gaba-v_i))/τ
        derivative[4] = a*(b_hat*v_e - w_e) + w_jump*r_e - a*b_hat
        derivative[5] = a*(b_hat*v_i - w_i) + w_jump*r_i - a*b_hat
        derivative[6] = -g_ampa/τ_AMPA + s_jump * r_e
        derivative[7] = -g_gaba/τ_GABA + s_jump * r_i
        return derivative











# -*- coding: utf-8 -*-

from ANNarchy import Neuron

Izhikevich_Hamker = Neuron(
parameters="""
    a = 0.02
    b = 0.2
    c = -72.0
    d = 6.0
    n0 = 140.
    n1 = 5.0
    n2 = 0.04
    I = 0.0
    tau_refrac = 10.0
    tau_ampa = 10.0
    tau_gaba = 10.0
    E_ampa = 0.0
    E_gaba = -90.0
    tau_syn = 1.0
    C = 1.0
    v_th = 30.0
    Vr = 0.0
""",
equations="""
    I_syn_ex = - g_ampa*(v-E_ampa)
    I_syn_in = - g_gaba*(v-E_gaba)
    I_syn = I_syn_ex + I_syn_in - g_base*v
    dg_base/dt = -g_base/tau_syn : init = 0
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u/C  + I + I_syn : init = -72.0
    du/dt = a*(b*(v-Vr)-u) : init = -14.4
""",
spike = """
    v>=v_th
""",
reset = """
    v = c
    u = u+d
""",
refractory = """tau_refrac"""
)

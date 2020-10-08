# -*- coding: utf-8 -*-

from ANNarchy import Neuron
from ANNarchy.extensions.bold import *  # importing NormProjection, ACCProjection and BoldMonitor


####################################################   NEURON MODELS   #################################################

Hybrid_neuron = Neuron(
parameters="""
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    n0 = 140.
    n1 = 5.0
    n2 = 0.04
    I = 0.0
    tau_ampa = 10
    tau_gaba = 10
    E_ampa = 0.0
    E_gaba = -90.0
    tau_syn = 1.0
    SteadyStateNoise = 0.0
    baseline = 0.0
    tau_noise=200.
    C = 1.0
""",
equations="""
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0
    dslowNoise/dt = (-slowNoise+SteadyStateNoise)/tau_noise : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u/C  + I - g_ampa*(v-E_ampa) - g_gaba*(v-E_gaba) + slowNoise + baseline : init = -70.
    du/dt = a*(b*(v)-u) : init = -18.55
    tau_syn*dsyn/dt = -syn 
""",
spike = """
    v>=30
""",
reset = """
    v = c
    u = u+d
""",
refractory = 10.0
)


Striatum_neuron = Neuron(
parameters="""
    a = 0.05       
    b = -20.0      
    c = -55.0      
    d = 377        
    n0 = 61.65119  
    n1 = 2.594639  
    n2 = 0.022799  
    I = 0.0        
    tau_ampa = 10  
    tau_gaba = 10  
    E_ampa = 0.0   
    E_gaba = -90.0 
    Vr = -80.      
    C  = 50.       
    tau_syn = 1.0
    SteadyStateNoise = 0.0
    tau_noise=200.
    baseline=0.0
""",
equations="""
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0 
    dslowNoise/dt = (-slowNoise+SteadyStateNoise)/tau_noise : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u/C  + I/C - g_ampa*(v-E_ampa) - g_gaba*(v-E_gaba)  + slowNoise + baseline : init = -70.
    du/dt = a*(b*(v-Vr)-u) : init = -18.55
    tau_syn*dsyn/dt = -syn 
""",
spike = """
    v>=40
""",
reset = """
    v = c
    u = u+d
""",
refractory = 10.0
)


BoldNeuron = Neuron(
parameters = """
    noise = 0.0
""",
equations = """
    r=sum(exc)+noise
"""
)


Poisson_neuron = Neuron(
parameters = """
    rates = 0.0
""",
equations = """
    p = Uniform(0.0, 1.0) * 1000.0 / dt
""",
spike = """
    p <= rates
    """,
reset = """
   p=0.0
"""
)

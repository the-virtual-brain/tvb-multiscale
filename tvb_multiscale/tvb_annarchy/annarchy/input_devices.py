# -*- coding: utf-8 -*-

from ANNarchy import Neuron


# The following devices current (rate) injection to spiking population
# follow the examples on Hybrid networks: https://annarchy.readthedocs.io/en/latest/manual/Hybrid.html


CurrentInjector = Neuron(
name="CurrentInjector",
equations="""
    r = amplitude
""",
parameters="""
    amplitude = 0.0
"""
)


DCCurrentInjector = CurrentInjector
DCCurrentInjector.name = "DCCurrentInjector"


ACCurrentInjector = Neuron(
name="ACCurrentInjector",
equations="""
    r = amplitude * sin(omega*t + phase) + offset
""",
parameters="""
    omega = 0.0
    amplitude = 1.0
    phase = 0.0
    offset = 0.0
"""
)


CurrentProxy = Neuron(
name="CurrentProxy",
equations="""
    r = sum(exc)
"""
)

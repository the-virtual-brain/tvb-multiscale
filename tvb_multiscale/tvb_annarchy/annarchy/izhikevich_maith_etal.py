# -*- coding: utf-8 -*-

from ANNarchy import *
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


# ###################################################   PARAMETERS   #####################################################
#
# """
# General Parameters.
# """
#
# params = {}
#
# ### General Parameters ###
# params['timestep'] = 0.1
# params['num_threads'] = 1
# # params['seedVal_simulations'] = 10
#
#
# ### General Model Parameters ###
# params['population_size'] = 200
# params['numberOfNeuronsCortex'] = 600
# params['noiseSYAC'] = 0.05
# params['GPinoiseSYAC'] = 0.0
# params['noiseFrequency'] = 1000.0  # every x ms the noise changes
#
# ### Regions Parameters ###
# ## Cortex
# params['popV1_a'] = 0.02
# params['popV1_b'] = 0.2
# params['popV1_c'] = -72
# params['popV1_d'] = 6
# params['popV1_I'] = 0.0
# params['Cortex_baseline'] = 50
# params['Cortex_noise_mean'] = 0
# params['Cortex_noise_sd'] = 10
# params['Cortex_noise_delta'] = 5
# params['popV1_v'] = -72
# params['popV1_u'] = -14
# params['popV1_C'] = 1.0
# params['popV1_n0'] = 140.0
# params['popV1_n1'] = 5.0
# params['popV1_n2'] = 0.04
#
# ## CortexI
# params['popInhibit_a'] = 0.02
# params['popInhibit_b'] = 0.2
# params['popInhibit_c'] = -72
# params['popInhibit_d'] = 6
# params['popInhibit_I'] = 0.0
# params['CortexInhib_baseline'] = 0
# # params['CortexInhib_noise_mean'] = 0 not used / CorI no noise
# # params['CortexInhib_noise_sd'] = 0 not used / CorI no noise
# # params['CortexInhib_noise_delta'] = 0 not used / CorI no noise
# params['popInhibit_v'] = -72
# params['popInhibit_u'] = -14
# params['popInhibit_C'] = 1.0
# params['popInhibit_n0'] = 140.0
# params['popInhibit_n1'] = 5.0
# params['popInhibit_n2'] = 0.04
#
# ## Striatum
# params['Str_a'] = 0.05
# params['Str_b'] = -20.0
# params['Str_c'] = -55.0
# params['Str_d'] = 377
# params['Str_I'] = 0.0
# params['Str_baseline'] = 0
# params['Str_noise_mean'] = 0
# params['Str_noise_sd'] = 2
# params['Str_noise_delta'] = 0
# params['Str_v'] = -70
# params['Str_u'] = -18.55
# params['Str_C'] = 50.0
# params['Str_n0'] = 61.65119
# params['Str_n1'] = 2.594639
# params['Str_n2'] = 0.022799
#
# ## GPi
# params['GPi_a'] = 0.005
# params['GPi_b'] = 0.585
# params['GPi_c'] = -65.
# params['GPi_d'] = 4.0
# params['GPi_I'] = 0.0
# params['GPi_baseline'] = 30
# params['GPi_noise_mean'] = 0
# params['GPi_noise_sd'] = 3
# params['GPi_noise_delta'] = 0
# params['GPi_v'] = -70
# params['GPi_u'] = -18.55
# params['GPi_C'] = 1.0
# params['GPi_n0'] = 140.0
# params['GPi_n1'] = 5.0
# params['GPi_n2'] = 0.04
#
# ## GPe
# params['GPe_a'] = 0.005
# params['GPe_b'] = 0.585
# params['GPe_c'] = -65.
# params['GPe_d'] = 4.0
# params['GPe_I'] = 0.0
# params['GPe_baseline'] = 12
# params['GPe_noise_mean'] = 0
# params['GPe_noise_sd'] = 5
# params['GPe_noise_delta'] = 0
# params['GPe_v'] = -70
# params['GPe_u'] = -18.55
# params['GPe_C'] = 1.0
# params['GPe_n0'] = 140.0
# params['GPe_n1'] = 5.0
# params['GPe_n2'] = 0.04
#
# ## STN
# params['STN_a'] = 0.005
# params['STN_b'] = 0.265
# params['STN_c'] = -65.
# params['STN_d'] = 2.0
# params['STN_I'] = 0.0
# params['STN_baseline'] = 3
# params['STN_noise_mean'] = 0
# params['STN_noise_sd'] = 2
# params['STN_noise_delta'] = 0
# params['STN_v'] = -70
# params['STN_u'] = -18.55
# params['STN_C'] = 1.0
# params['STN_n0'] = 140.0
# params['STN_n1'] = 5.0
# params['STN_n2'] = 0.04
#
# ## Thalamus
# params['Thalamus_a'] = 0.02
# params['Thalamus_b'] = 0.25
# params['Thalamus_c'] = -65.
# params['Thalamus_d'] = 0.05
# params['Thalamus_I'] = 0.0
# params['Thalamus_baseline'] = 3.5
# params['Thalamus_noise_mean'] = 0
# params['Thalamus_noise_sd'] = 2
# params['Thalamus_noise_delta'] = 0
# params['Thalamus_v'] = -70
# params['Thalamus_u'] = -18.55
# params['Thalamus_C'] = 1.0
# params['Thalamus_n0'] = 140.0
# params['Thalamus_n1'] = 5.0
# params['Thalamus_n2'] = 0.04
#
# ### Parameters to be fitted (here "standard" values) ###
# ## Weights
# # Cortical Input
# params['CSD1_weights'] = 0.005
# params['CSD2_weights'] = 0.005
# params['CSTN_weights'] = 0.015
# # BG Input
# params['D1GPi_weights'] = 0.2
# params['D2GPe_weights'] = 0.2
# params['STNGPe_weights'] = 0.2
# params['STNGPi_weights'] = 0.2
# # Inner BG
# params['GPeSTN_weights'] = 0.002
# params['GPeGPi_weights'] = 0.03
# params['GPiThal_weights'] = 0.013
# # Feedback
# params['ThalSD2_weights'] = 0.002
# params['ThalSD1_weights'] = 0.002
# # Laterals
# params['GPeGPe_weights'] = 0.009
# params['GPiGPi_weights'] = 0.009
# params['SD1SD1_weights'] = 0.0025
# params['SD2SD2_weights'] = 0.0025
# params['STNSTN_weights'] = 0.0
# # Cortex E-I Loop
# params['V1Inh_weights'] = 0.0337
# params['InhV1_weights'] = 0.3
# params['InhInh_weights'] = 0.4
#
# ## Probs
# # Cortical Input
# params['CSD1_probs'] = 0.1
# params['CSD2_probs'] = 0.1
# params['CSTN_probs'] = 0.1
# # BG Input
# params['D1GPi_probs'] = 0.25
# params['D2GPe_probs'] = 0.25
# params['STNGPe_probs'] = 0.25
# params['STNGPi_probs'] = 0.25
# # Inner BG
# params['GPeSTN_probs'] = 0.25
# params['GPeGPi_probs'] = 0.35
# params['GPiThal_probs'] = 0.2
# # Feedback
# params['ThalSD2_probs'] = 0.4
# params['ThalSD1_probs'] = 0.4
# # Laterals
# params['GPeGPe_probs'] = 0.35
# params['GPiGPi_probs'] = 0.35
# params['SD1SD1_probs'] = 0.3
# params['SD2SD2_probs'] = 0.3
# params['STNSTN_probs'] = 0.0
# # Cortex E-I Loop
# params['V1Inh_probs'] = 0.4
# params['InhV1_probs'] = 0.4
# params['InhInh_probs'] = 0.4
#
#
#
# ######################################################   FUNCTIONS   ###################################################
# from ANNarchy.extensions.bold import *
# import random
# import sys
# import pickle
#
#
# def multiplyWeights(proj, factor):
#     weights = net.get(proj).w
#     weightsArray = np.array(net.get(proj).w)
#     weightsArray *= factor
#     net.get(proj).w = weightsArray.tolist()
#
#
# def update_probs(net, name, val):
#     """
#     Update the connection probability
#     """
#     t = net.get_projection(name)._connection_args
#     l = list(t)
#     l[0] = val
#     net.get_projection(name)._connection_args = tuple(l)
#
#
# def update_weights(net, name, val):
#     """
#     Update the default weight
#     """
#     t = net.get_projection(name)._connection_args
#     l = list(t)
#     l[1] = val
#     net.get_projection(name)._connection_args = tuple(l)
#
#
# def simulateNoisePeriods(net, duration, loadNoise=False, saveNoise=False):
#     try:
#         if duration % params['noiseFrequency'] != 0:
#             raise ValueError('duration (' + str(duration) + ') has to be a multiple of noiseFrequency')
#     except ValueError as ve:
#         print(ve)
#         sys.exit(1)
#
#     if loadNoise:
#         with open("noise_for_simulations_duration_" + str(int(duration)) + ".txt", "rb") as f:
#             noiseValues = pickle.load(f)
#     else:
#         noiseValues = []
#
#     steps = int(duration / params['noiseFrequency'])
#     for NoisePeriod in range(steps):
#         if loadNoise:
#             [V1_noise, SD1_noise, SD2_noise, GPi_noise, GPe_noise, STN_noise, Thalamus_noise, GPiSyn_noise,
#              GPeSyn_noise, STNSyn_noise, StrBothSyn_noise, ThalSyn_noise, CortexSyn_noise] = noiseValues[NoisePeriod]
#         else:
#             V1_noise = np.random.normal(params['Cortex_noise_mean'] + np.random.uniform(-params['Cortex_noise_delta'],
#                                                                                         params['Cortex_noise_delta']),
#                                         params['Cortex_noise_sd'], net.get_population('V1').geometry)
#             SD1_noise = np.random.normal(
#                 params['Str_noise_mean'] + np.random.uniform(-params['Str_noise_delta'], params['Str_noise_delta']),
#                 params['Str_noise_sd'], net.get_population('SD1').geometry)
#             SD2_noise = np.random.normal(
#                 params['Str_noise_mean'] + np.random.uniform(-params['Str_noise_delta'], params['Str_noise_delta']),
#                 params['Str_noise_sd'], net.get_population('SD2').geometry)
#             GPi_noise = np.random.normal(
#                 params['GPi_noise_mean'] + np.random.uniform(-params['GPi_noise_delta'], params['GPi_noise_delta']),
#                 params['GPi_noise_sd'], net.get_population('GPi').geometry)
#             GPe_noise = np.random.normal(
#                 params['GPe_noise_mean'] + np.random.uniform(-params['GPe_noise_delta'], params['GPe_noise_delta']),
#                 params['GPe_noise_sd'], net.get_population('GPe').geometry)
#             STN_noise = np.random.normal(
#                 params['STN_noise_mean'] + np.random.uniform(-params['STN_noise_delta'], params['STN_noise_delta']),
#                 params['STN_noise_sd'], net.get_population('STN').geometry)
#             Thalamus_noise = np.random.normal(
#                 params['Thalamus_noise_mean'] + np.random.uniform(-params['Thalamus_noise_delta'],
#                                                                   params['Thalamus_noise_delta']),
#                 params['Thalamus_noise_sd'], net.get_population('Thalamus').geometry)
#             GPiSyn_noise = np.random.uniform(0, 1, net.get_population('GPiSynA').geometry) * params['GPinoiseSYAC']
#             GPeSyn_noise = np.random.uniform(0, 1, net.get_population('GPeSynA').geometry) * params['noiseSYAC']
#             STNSyn_noise = np.random.uniform(0, 1, net.get_population('STNSynA').geometry) * params['noiseSYAC']
#             StrBothSyn_noise = np.random.uniform(0, 1, net.get_population('SrtBothSynA').geometry) * params['noiseSYAC']
#             ThalSyn_noise = np.random.uniform(0, 1, net.get_population('ThalSynA').geometry) * params['noiseSYAC']
#             CortexSyn_noise = np.random.uniform(0, 1, net.get_population('CortexSynA').geometry) * params['noiseSYAC']
#
#         net.get_population('V1').SteadyStateNoise = V1_noise
#         net.get_population('SD1').SteadyStateNoise = SD1_noise
#         net.get_population('SD2').SteadyStateNoise = SD2_noise
#         net.get_population('GPi').SteadyStateNoise = GPi_noise
#         net.get_population('GPe').SteadyStateNoise = GPe_noise
#         net.get_population('STN').SteadyStateNoise = STN_noise
#         net.get_population('Thalamus').SteadyStateNoise = Thalamus_noise
#         net.get_population('GPiSynA').noise = GPiSyn_noise
#         net.get_population('GPeSynA').noise = GPeSyn_noise
#         net.get_population('STNSynA').noise = STNSyn_noise
#         net.get_population('SrtBothSynA').noise = StrBothSyn_noise
#         net.get_population('ThalSynA').noise = ThalSyn_noise
#         net.get_population('CortexSynA').noise = CortexSyn_noise
#         net.simulate(params['noiseFrequency'])
#
#         if saveNoise:
#             noiseValues.append(
#                 [V1_noise, SD1_noise, SD2_noise, GPi_noise, GPe_noise, STN_noise, Thalamus_noise, GPiSyn_noise,
#                  GPeSyn_noise, STNSyn_noise, StrBothSyn_noise, ThalSyn_noise, CortexSyn_noise])
#
#     if saveNoise:
#         with open("noise_for_simulations_duration_" + str(int(duration)) + ".txt", "wb") as f:
#             pickle.dump(noiseValues, f)

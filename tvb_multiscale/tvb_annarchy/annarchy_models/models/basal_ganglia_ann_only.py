import ANNarchy as ann
from tvb_multiscale.tvb_annarchy.annarchy.models import Hybrid_neuron, Striatum_neuron

ann.clear()
ann.setup(dt=0.025)

# Define populations

popGPe_Left_I = ann.Population(200, neuron=Hybrid_neuron, name="I")
popGPe_Left_I.I = 12.0
popGPe_Left_I.a = 0.005
popGPe_Left_I.b = 0.585
popGPe_Left_I.c = -65.0
popGPe_Left_I.d = 4.0

popGPe_Right_I = ann.Population(200, neuron=Hybrid_neuron, name="I")
popGPe_Right_I.I = 12.0
popGPe_Right_I.a = 0.005
popGPe_Right_I.b = 0.585
popGPe_Right_I.c = -65.0
popGPe_Right_I.d = 4.0

popGPi_Left_I = ann.Population(200, neuron=Hybrid_neuron, name="I")
popGPi_Left_I.I = 30.0
popGPi_Left_I.a = 0.005
popGPi_Left_I.b = 0.585
popGPi_Left_I.c = -65.0
popGPi_Left_I.d = 4.0

popGPi_Right_I = ann.Population(200, neuron=Hybrid_neuron, name="I")
popGPi_Right_I.I = 30.0
popGPi_Right_I.a = 0.005
popGPi_Right_I.b = 0.585
popGPi_Right_I.c = -65.0
popGPi_Right_I.d = 4.0

popSTN_Left_E = ann.Population(200, neuron=Hybrid_neuron, name="E")
popSTN_Left_E.a = 0.005
popSTN_Left_E.b = 0.265
popSTN_Left_E.c = -65.0
popSTN_Left_E.d = 2.0
popSTN_Left_E.I = 3.0

popSTN_Right_E = ann.Population(200, neuron=Hybrid_neuron, name="E")
popSTN_Right_E.a = 0.005
popSTN_Right_E.b = 0.265
popSTN_Right_E.c = -65.0
popSTN_Right_E.d = 2.0
popSTN_Right_E.I = 3.0

popThal_Left_E = ann.Population(200, neuron=Hybrid_neuron, name="E")
popThal_Left_E.a = 0.02
popThal_Left_E.b = 0.25
popThal_Left_E.c = -65.0
popThal_Left_E.d = 0.05
popThal_Left_E.I = 3.5

popThal_Right_E = ann.Population(200, neuron=Hybrid_neuron, name="E")
popThal_Right_E.a = 0.02
popThal_Right_E.b = 0.25
popThal_Right_E.c = -65.0
popThal_Right_E.d = 0.05
popThal_Right_E.I = 3.5

popStr_Left_I1 = ann.Population(200, neuron=Striatum_neuron, name="I1")
popStr_Left_I2 = ann.Population(200, neuron=Striatum_neuron, name="I2")
popStr_Right_I1 = ann.Population(200, neuron=Striatum_neuron, name="I1")
popStr_Right_I2 = ann.Population(200, neuron=Striatum_neuron, name="I2")

# Intra-region projections
# as in builder, only inhibitory

projGPe_Left_I = ann.Projection(popGPe_Left_I, popGPe_Left_I, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                         allow_self_connections=True,
                                                                                         force_multiple_weights=False)
projGPe_Right_I = ann.Projection(popGPe_Right_I, popGPe_Right_I, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                            allow_self_connections=True,
                                                                                            force_multiple_weights=False)
projGPi_Left_I = ann.Projection(popGPi_Left_I, popGPi_Left_I, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                         allow_self_connections=True,
                                                                                         force_multiple_weights=False)
projGPi_Right_I = ann.Projection(popGPi_Right_I, popGPi_Right_I, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                            allow_self_connections=True,
                                                                                            force_multiple_weights=False)
projStr_Left_I1 = ann.Projection(popStr_Left_I1, popStr_Left_I1, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                            allow_self_connections=True,
                                                                                            force_multiple_weights=False)
projStr_Left_I2 = ann.Projection(popStr_Left_I2, popStr_Left_I2, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                            allow_self_connections=True,
                                                                                            force_multiple_weights=False)
projStr_Right_I1 = ann.Projection(popStr_Right_I1, popStr_Right_I1, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                               allow_self_connections=True,
                                                                                               force_multiple_weights=False)
projStr_Right_I2 = ann.Projection(popStr_Right_I2, popStr_Right_I2, 'gaba').connect_all_to_all(weights=1.0, delays=0.05,
                                                                                               allow_self_connections=True,
                                                                                               force_multiple_weights=False)

# Connections between regions

# Projection 6 -> 2
projStr_I1_GPi_Left = ann.Projection(popStr_Left_I1, popGPi_Left_I, 'gaba', name="I1 -> I").connect_all_to_all(
    weights=0.078125, delays=4.237235961892867,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 6 -> 3, 7 -> 2 left out (weight = 0)
# Projection 7 -> 3
projStr_I1_GPi_Right = ann.Projection(popStr_Left_I1, popGPi_Left_I, 'gaba', name="I1 -> I").connect_all_to_all(
    weights=0.078125, delays=4.4414688584867665,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 6["I2"] -> 0
projStr_I2_GPe_Left = ann.Projection(popStr_Left_I2, popGPe_Left_I, 'gaba', name="I2 -> I").connect_all_to_all(
    weights=0.078125, delays=2.8642076225821302,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 6["I2"] -> 1, 7["I2"] -> 0 left out (weight = 0)
# Projection 7["I2"] -> 1
projStr_I2_GPe_Right = ann.Projection(popStr_Right_I2, popGPe_Right_I, 'gaba', name="I2 -> I").connect_all_to_all(
    weights=0.078125, delays=2.94876594895625,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 0 -> 2
projGPe_GPi_Left = ann.Projection(popGPe_Left_I, popGPi_Left_I, 'gaba', name="I -> I").connect_all_to_all(
    weights=0.078125, delays=1.5412287562752869,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 0 -> 3, 1 -> 2 left out (weight = 0)
# Projection 1 -> 3
projGPe_GPi_Right = ann.Projection(popGPe_Right_I, popGPi_Right_I, 'gaba', name="I -> I").connect_all_to_all(
    weights=0.078125, delays=1.6136932578069734,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 2 -> 8
projGPi_Thal_Left = ann.Projection(popGPi_Left_I, popThal_Left_E, 'gaba', name="I -> E").connect_all_to_all(
    weights=0.078125, delays=5.718029735236066,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 2 -> 9, 3 -> 8 left out (weight = 0)
# Projection 3 -> 9
projGPi_Thal_Right = ann.Projection(popGPi_Right_I, popThal_Right_E, 'gaba', name="I -> E").connect_all_to_all(
    weights=0.078125, delays=6.2724064499556995,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 0 -> 4
projGPe_STN_Left = ann.Projection(popGPe_Left_I, popSTN_Left_E, 'gaba', name="I -> E").connect_all_to_all(
    weights=0.078125, delays=11.202326146081633,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 0 -> 5, 1 -> 4 left out (weight = 0)
# Projection 1 -> 5
projGPe_STN_Right = ann.Projection(popGPe_Right_I, popSTN_Right_E, 'gaba', name="I -> E").connect_all_to_all(
    weights=0.078125, delays=5.1992490412363335,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 8 -> 6["I1"]
projThal_Str_I1_Left = ann.Projection(popThal_Left_E, popStr_Left_I1, 'ampa', name="E -> I1").connect_all_to_all(
    weights=0.078125, delays=7.9678738339179995,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 8 -> 6["I2"]
projThal_Str_I2_Left = ann.Projection(popThal_Left_E, popStr_Left_I2, 'ampa', name="E -> I2").connect_all_to_all(
    weights=0.078125, delays=7.9678738339179995,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 8 -> 7["I1"], 8 -> 7["I2"], 9 -> 6["I1"], 9 -> 6["I2"]  left out (weight = 0)
# Projection 9 -> 7["I1"]
projThal_Str_I1_Right = ann.Projection(popThal_Right_E, popStr_Right_I1, 'ampa', name="E -> I1").connect_all_to_all(
    weights=0.006463630506447433, delays=8.8572614170515,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 9 -> 7["I2"]
projThal_Str_I2_Right = ann.Projection(popThal_Right_E, popStr_Right_I2, 'ampa', name="E -> I2").connect_all_to_all(
    weights=0.006463630506447433, delays=8.8572614170515,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 4 -> 0
projSTN_GPe_Left = ann.Projection(popSTN_Left_E, popGPe_Left_I, 'ampa', name="E -> I").connect_all_to_all(
    weights=0.078125, delays=11.202326146081633,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 4 -> 1  left out (weight = 0)
# Projection 4 -> 2
projSTN_GPi_Left = ann.Projection(popSTN_Left_E, popGPi_Left_I, 'ampa', name="E -> I").connect_all_to_all(
    weights=0.078125, delays=10.168185226479666,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 4 -> 3, 5 -> 0  left out (weight = 0)
# Projection 5 -> 1
projSTN_GPe_Right = ann.Projection(popSTN_Right_E, popGPe_Right_I, 'ampa', name="E -> I").connect_all_to_all(
    weights=0.078125, delays=5.1992490412363335,
    allow_self_connections=True,
    force_multiple_weights=False)
# Projection 5 -> 2 left out (weight = 0)
# Projection 5 -> 3
projSTN_GPi_Right = ann.Projection(popSTN_Right_E, popGPi_Right_I, 'ampa', name="E -> I").connect_all_to_all(
    weights=0.078125, delays=3.724878156355367,
    allow_self_connections=True,
    force_multiple_weights=False)

# Output devices / monitors
# Excitatory spikes
mSTN_Left_spike = ann.Monitor(popSTN_Left_E, ["spike"])
mSTN_Right_spike = ann.Monitor(popSTN_Right_E, ["spike"])
mThal_Left_spike = ann.Monitor(popThal_Left_E, ["spike"])
mThal_Right_spike = ann.Monitor(popThal_Right_E, ["spike"])

# Inhibitory spikes
mGPe_Left_spike = ann.Monitor(popGPe_Left_I, ["spike"])
mGPe_Right_spike = ann.Monitor(popGPe_Right_I, ["spike"])
mGPi_Left_spike = ann.Monitor(popGPi_Left_I, ["spike"])
mGPi_Right_spike = ann.Monitor(popGPi_Right_I, ["spike"])
mStr_Left_I1_spike = ann.Monitor(popStr_Left_I1, ["spike"])
mStr_Right_I1_spike = ann.Monitor(popStr_Right_I1, ["spike"])
mStr_Left_I2_spike = ann.Monitor(popStr_Left_I2, ["spike"])
mStr_Right_I2_spike = ann.Monitor(popStr_Right_I2, ["spike"])

# Other variables, excitatory populations
mSTN_Left = ann.Monitor(popSTN_Left_E, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mSTN_Right = ann.Monitor(popSTN_Right_E, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mThal_Left = ann.Monitor(popThal_Left_E, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mThal_Right = ann.Monitor(popThal_Right_E, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)

# Other variables, inhibitory populations
mGPe_Left = ann.Monitor(popGPe_Left_I, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mGPe_Right = ann.Monitor(popGPe_Right_I, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mGPi_Left = ann.Monitor(popGPi_Left_I, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mGPi_Right = ann.Monitor(popGPi_Right_I, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mStr_Left_I1 = ann.Monitor(popStr_Left_I1, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mStr_Right_I1 = ann.Monitor(popStr_Right_I1, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mStr_Left_I2 = ann.Monitor(popStr_Left_I2, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)
mStr_Right_I2 = ann.Monitor(popStr_Right_I2, ["v", "u", "syn", "g_ampa", "g_gaba"], period=1.0)

# Inputs

inpEstn = ann.PoissonPopulation(geometry=200, name="BaselineEstn", rates=500.0)
projInpEstnLeft = ann.Projection(inpEstn, popSTN_Left_E, 'exc')
#projInpEstnLeft.connect_one_to_one(weights=0.009, delays=0) #this produces segmentation fault!
projInpEstnLeft.connect_one_to_one(weights=0.009)
#projInpEstnRight = ann.Projection(inpEstn, popSTN_Left_E, 'exc')

ann.compile()
ann.simulate(100)

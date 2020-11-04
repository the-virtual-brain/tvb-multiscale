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

###
'''
home_path = os.path.join(os.getcwd().split("tvb-multiscale")[0], "tvb-multiscale")
DATA_PATH = os.path.join(home_path, "examples/data")
w = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/conn_denis_weights.txt"))
c = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/aal_plus_BG_centers.txt"), usecols=range(1, 3))
rl = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/aal_plus_BG_centers.txt"), dtype="str", usecols=(0,))
t = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/BGplusAAL_tract_lengths.txt"))


tvb_sim_model = ReducedWongWangExcIO
annarchy_model_builder = BasalGangliaIzhikevichBuilder
tvb_annarchy_builder = None
annarchy_populations_order = 200
tvb_to_annarchy_mode = None
annarchy_to_tvb = False
exclusive_nodes = True
connectivity = Connectivity(region_labels=rl, weights=w, centres=c, tract_lengths=t)
delays_flag = True
simulation_length = 110.0
transient = 0.0
variables_of_interest = None
config = None
plot_write = True

annarchy_nodes_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# Remove BG -> Cortex connections
w[[0, 1, 2, 3, 6, 7], :][:, 10:] = 0.0

model_params = {}
import faulthandler
faulthandler.enable()

if config is None:
    config = Config(
                output_base=results_path_fun(annarchy_model_builder, tvb_annarchy_builder, tvb_to_annarchy_mode,
                                             annarchy_to_tvb, config))

plotter = Plotter(config)


# ------2. Build the annarchy network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

print("Building annarchy network...")
tic = time.time()

# Build a annarchy network model with the corresponding builder
# Using all default parameters for this example

#annarchy_model_builder = annarchy_model_builder(simulator, annarchy_nodes_ids, config=config)
#annarchy_model_builder.population_order = annarchy_populations_order

default_population = {"model": config.DEFAULT_MODEL, "scale": 1, "params": {}, "nodes": None}
default_population["model"] = Hybrid_neuron

# Common order of neurons' number per population:
population_order = 200

params_common = {"E_ampa": 0.0, "E_gaba": -90.0, "v_th": 30.0, "c": -65.0,
                      "C": 1.0, "I": 0.0,
                      "tau_refrac": 10.0, "tau_syn": 1.0, "tau_ampa": 10.0, "tau_gaba": 10.0,
                      "n0": 140.0, "n1": 5.0, "n2": 0.04}
from copy import deepcopy
paramsI = deepcopy(params_common)
paramsI.update({"a": 0.005, "b": 0.585, "d": 4.0})

paramsE = deepcopy(params_common)
paramsStr = deepcopy(params_common)
paramsStr.update({"v_th": 40.0, "C": 50.0,
                       "n0": 61.65119, "n1": 2.594639, "n2": 0.022799,
                       "a": 0.05, "b": -20.0, "c": -55.0, "d": 377.0})

Igpe_nodes_ids = [0, 1]
Igpi_nodes_ids = [2, 3]
Estn_nodes_ids = [4, 5]
Eth_nodes_ids = [8, 9]
Istr_nodes_ids = [6, 7]

Estn_stim = {"rate": 500.0, "weight": 0.009}
Igpe_stim = {"rate": 100.0, "weight": 0.015}
Igpi_stim = {"rate": 700.0, "weight": 0.02}

populations = [
    {"label": "E", "model": default_population["model"],  # Estn in [4, 5], Eth in [8, 9]
     "params": paramsE, "nodes": Estn_nodes_ids + Eth_nodes_ids,  # None means "all"
     "scale": 1.0},
    {"label": "I", "model": default_population["model"],  # Igpe in [0, 1], Igpi in [2, 3]
     "params": paramsI, "nodes": Igpe_nodes_ids + Igpi_nodes_ids,  # None means "all"
     "scale": 1.0},
    {"label": "I1", "model": Striatum_neuron,  # Isd1 in [6, 7]
     "params": paramsStr, "nodes": Istr_nodes_ids,  # None means "all"
     "scale": 1.0},
    {"label": "I2", "model": Striatum_neuron,  # Isd2 in [6, 7]
     "params": paramsStr, "nodes": Istr_nodes_ids,  # None means "all"
     "scale": 1.0}
]
from collections import OrderedDict
populations_sizes = OrderedDict()
for pop in populations:
    populations_sizes[pop["label"]] = int(np.round(pop["scale"] * population_order))

synapse_model = default_populations_connection["synapse_model"]  # "DefaultSpikingSynapse"
# default connectivity spec:
# conn_spec= {'method': "all_to_all"}
conn_spec = self.default_populations_connection["conn_spec"]
conn_spec["allow_self_connections"] = True
conn_spec["force_multiple_weights"] = False

# Intra-regions'-nodes' connections
self.populations_connections = []
for pop in self.populations:
    # Only self-connections and only for all inhibitory  populations
    if pop["label"][0] == "I":
        self.populations_connections.append(
            {"source": pop["label"], "target": pop["label"],
             "synapse_model": synapse_model, "conn_spec": conn_spec,
             "weight": 1.0, "delay": self.default_min_delay,  # 0.001
             "receptor_type": "gaba", "nodes": pop["nodes"]})

# NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
self.global_coupling_scaling = tvb_simulator.coupling.a[0].item()
# if we use Reduced Wong Wang model, we also need to multiply with the global coupling constant G:
self.global_coupling_scaling *= tvb_simulator.model.G[0].item()

# Inter-regions'-nodes' connections
self.nodes_connections = []
for src_pop, trg_pop, src_nodes, trg_nodes in \
        zip(
            # "Isd1->Igpi", "Isd2->Igpe", "Igpe->Igpi", "Igpi->Eth", "Igpe->Estn", "Eth->[Isd1, Isd2]", "Estn->[Igpe, Igpi]",
            ["I1", "I2", "I", "I", "I", "E", "E"],  # source
            ["I", "I", "I", "E", "E", ["I1", "I2"], "I"],  # target
            [[6, 7], [6, 7], [0, 1], [2, 3], [0, 1], [8, 9], [4, 5]],  # source nodes
            [[2, 3], [0, 1], [2, 3], [8, 9], [4, 5], [6, 7], [0, 1, 2, 3]]):  # target nodes
    if src_pop[0] == "I":
        target = "gaba"
    else:
        target = "ampa"
    self.nodes_connections.append(
        {"source": src_pop, "target": trg_pop,
         "synapse_model": self.default_nodes_connection["synapse_model"],
         "conn_spec": self.default_nodes_connection["conn_spec"],
         "weight": TVBWeightFun(self.tvb_weights, self.global_coupling_scaling),
         "delay": lambda source_node, target_node: self.tvb_delay_fun(source_node, target_node),
         "receptor_type": target, "source_nodes": src_nodes, "target_nodes": trg_nodes})

# Creating  devices to be able to observe ANNarchy activity:
self.output_devices = []
params = self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["SpikeMonitor"]
for pop in self.populations:
    connections = OrderedDict({})
    #                      label <- target population
    params["label"] = pop["label"] + "_spikes"
    connections[params["label"]] = pop["label"]
    self.output_devices.append(
        {"model": "SpikeMonitor", "params": deepcopy(params),
         "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

# Labels have to be different for every connection to every distinct population
# params for baladron implementation commented out for the moment
# TODO: use baladron neurons
params = self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["Monitor"]
params.update({"period": 1.0, 'record_from': ["v", "u", "syn", "g_ampa", "g_gaba"]})
for pop in self.populations:
    connections = OrderedDict({})
    #               label    <- target population
    connections[pop["label"]] = pop["label"]
    params["label"] = pop["label"]
    self.output_devices.append(
        {"model": "Monitor", "params": deepcopy(params),
         "connections": connections, "nodes": pop["nodes"]})  # None means apply to all

# Create a spike stimulus input device
self.input_devices = [
    {"model": "PoissonPopulation",
     "params": {"rates": self.Estn_stim["rate"], "geometry": populations_sizes["E"], "name": "BaselineEstn"},
     "connections": {"BaselineEstn": ["E"]},  # "Estn"
     "nodes": self.Estn_nodes_ids,  # None means apply to all
     "weights": self.Estn_stim["weight"], "delays": 0.0, "receptor_type": "base"},
    {"model": "PoissonPopulation",
     "params": {"rates": self.Igpe_stim["rate"], "geometry": populations_sizes["I"], "name": "BaselineIgpe"},
     "connections": {"BaselineIgpe": ["I"]},  # "Igpe"
     "nodes": self.Igpe_nodes_ids,  # None means apply to all
     "weights": self.Igpe_stim["weight"], "delays": 0.0, "receptor_type": "base"},
    {"model": "PoissonPopulation",
     "params": {"rates": self.Igpi_stim["rate"], "geometry": populations_sizes["I"], "name": "BaselineIgpi"},
     "connections": {"BaselineIgpi": ["I"]},  # "Igpi"
     "nodes": self.Igpi_nodes_ids,  # None means apply to all
     "weights": self.Igpi_stim["weight"], "delays": 0.0, "receptor_type": "base"},
    # {"model": "ACCurrentInjector",
    #  "params": {"frequency": 30.0, "phase": 0.0, "amplitude": 1.0, "offset": 0.0},
    #  "connections": {"DBS_Estn": ["E"]},  # "Estn"
    #  "nodes": self.Estn_nodes_ids,  # None means apply to all
    #  "weights": 1.0, "delays": 0.0}
]

populations = []
populations_sizes = []
for pop in annarchy_model_builder.populations:
    populations.append(pop["label"])
    populations_sizes.append(int(np.round(pop["scale"] * annarchy_model_builder.population_order)))
# Common order of neurons' number per population:
annarchy_network = annarchy_model_builder.build_spiking_network()
annarchy_network.configure()
print(annarchy_network.print_str(connectivity=True))
print("Done! in %f min" % ((time.time() - tic) / 60))


    # -----------------------------------4. Simulate and gather results---------------------------------------------
print("Simulating ANNarchy only...")
t_start = time.time()
annarchy_network.Run(simulation_length)
print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
if plot_write:
    try:
        plot_write_results(results, simulator, populations=populations, populations_sizes=populations_sizes,
                           transient=transient, tvb_state_variable_type_label="State Variables",
                           tvb_state_variables_labels=simulator.model.variables_of_interest,
                           plotter=plotter, config=config)
    except Exception as e:
        print("Error in plotting or writing to files!:\n%s" % str(e))

'''

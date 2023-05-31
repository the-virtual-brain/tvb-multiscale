# import os
# os.chdir('examples/tvb_netpyne/notebooks')



from IPython.display import Image, display
import os
from collections import OrderedDict
import time
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import *

work_path = "/home/docker/packages/tvb-multiscale/examples/tvb_netpyne/notebooks" #os.getcwd()
data_path = os.path.expanduser("~/packages/tvb-multiscale/examples/data/basal_ganglia")
fit_data_path = os.path.join(data_path, "ANNarchyFittedModels/dataFits_2020_02_05/databestfits", )
data_mode = "patient" # "control", or "patient"
control_data = os.path.join(fit_data_path, "controlleft/OutputSim_Patient08.mat")
patient_data = os.path.join(fit_data_path, "patientleft/OutputSim_Patient09.mat")

INTERFACE_COUPLING_MODE = "spikeNet" # "TVB"  or "spikeNet"

if data_mode == "patient":
    subject_data = patient_data
    if INTERFACE_COUPLING_MODE == "TVB":
        TC_factor = 20.5 * 1e-2  # 0.205 
    else:
        TC_factor = 20.5 * 1e-2  # 0.205 
else:
    subject_data = control_data
    if INTERFACE_COUPLING_MODE == "TVB":
        TC_factor = 20.5 * 1e-2  # 0.205 
 
    else:
        TC_factor = 20.5 * 1e-2  # 0.205 
  
    
simulation_length = 150.0
transient = 25.0
start_stimulus = 100.0
init_cond_jitter = 0.0

SPIKING_NODES_DELAYS = False

simulation_mode = "rs"         # "stim" or "rs"
stim_target = "PY_pop"            #     "PY_pop",        "TC_pop"
stim_mode = "simple"           # "bi"  | "mono" | "simple"
                               # -------------------------
stim_freq = 0.0                # 130.0 |  120.0 |    0.0 
stim_ampl = -10.0              #  20.0 |  -35.0 |  -10.0 
stim_duration = 0.0            #   0.3 |    0.3 |    0.0
if simulation_mode == "stim":
    simulation_mode = simulation_mode + "_%s_%s" % (stim_target, stim_mode)

outputs_path = os.path.join(work_path, "outputs")
sim_mode_path = os.path.join(outputs_path, "TVBcortex_%s_coupl" % INTERFACE_COUPLING_MODE, 
                             data_mode, simulation_mode)
config = Config(output_base=sim_mode_path)
config.figures.SHOW_FLAG = True 
config.figures.SAVE_FLAG = True
config.figures.FIG_FORMAT = 'png'
config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
FIGSIZE = config.figures.DEFAULT_SIZE

from tvb_multiscale.core.plot.plotter import Plotter
plotter = Plotter(config.figures)

# For interactive plotting:
# %matplotlib notebook  

# Otherwise:
#%matplotlib inline 

from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO

# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the simulator by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold, EEG

# Load full TVB connectome connectivity

conn_path = os.path.join(data_path, "conn")

#Load AAL atlas normative connectome including the Basal Ganglia regions from Petersen et al. atlas
wTVB = np.loadtxt(os.path.join(conn_path, "conn_denis_weights.txt"))
cTVB = np.loadtxt(os.path.join(conn_path, "aal_plus_BG_centers.txt"),usecols=range(1,4))
rlTVB = np.loadtxt(os.path.join(conn_path, "aal_plus_BG_centers.txt"),dtype="str", usecols=(0,))
tlTVB = np.loadtxt(os.path.join(conn_path, "BGplusAAL_tract_lengths.txt"))

number_of_regions = len(rlTVB)
speed = np.array([4.0])
min_tt = speed.item() * 0.1
sliceTC = [8,9]
sliceCortex = slice(10, number_of_regions)

connTVB = Connectivity(region_labels=rlTVB, weights=wTVB, centres=cTVB, tract_lengths=tlTVB, speed=speed)

# Normalize connectivity weights
connTVB.weights = connTVB.scaled_weights(mode="region")
connTVB.weights /= np.percentile(connTVB.weights, 99)

sliceTCnet = slice(8,10) #BG (0,5) from both hemispheres = (0,10). only thalamus (bilateral) = 8&9

connTVBTC = Connectivity(region_labels=connTVB.region_labels[sliceTCnet], 
                             centres=connTVB.centres[sliceTCnet],
                             weights=connTVB.weights[sliceTCnet][:, sliceTCnet],
                             tract_lengths=connTVB.tract_lengths[sliceTCnet][:, sliceTCnet], 
                            speed=connTVB.speed)
connTVBTC.configure()

# plotter.plot_tvb_connectivity(connTVBleftBG);

# Construct the final connectivity to use for simulation:
connectivity = Connectivity(region_labels=connTVB.region_labels, 
                            centres=connTVB.centres,
                            weights=connTVB.weights,
                            tract_lengths=connTVB.tract_lengths, 
                            speed=connTVB.speed)

#connectivity = connTVB
connectivity.configure()

plotter.plot_tvb_connectivity(connectivity);

import matplotlib.pyplot as plt

# CHECK CONNECTOME

# plot connectome weights & tract lengths
f = plt.matshow(wTVB)
plt.savefig(sim_mode_path+"/figs/"+"connectivity.png")
f = plt.matshow(connTVB.tract_lengths)
plt.savefig(sim_mode_path+"/figs/"+"tracts_lengths.png")


from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial as CoSimulator

#white_matter_coupling = coupling.Linear(a=0.014)
# Create a TVB simulator and set all desired inputs
# (connectivity, model, surface, stimuli etc)
# We choose all defaults in this example
simulator = CoSimulator()
#simulator.use_numba = False
model_params = {"G": np.array([15.0])}
simulator.model = ReducedWongWangExcIO(**model_params)

simulator.connectivity = connectivity

simulator.integrator = HeunStochastic()
simulator.integrator.dt = 0.1
simulator.integrator.noise.nsig = np.array([1e-4])  # 1e-5

mon_raw = Raw(period=1.0)  # ms
simulator.monitors = (mon_raw, )
simulator.connectivity.set_idelays(simulator.integrator.dt)
simulator.initial_conditions = None

plotter.plot_tvb_connectivity(simulator.connectivity);


# # Serializing TVB cosimulator is necessary for parallel cosimulation:
# from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
# from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
# sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
# sim_serial = serialize_tvb_cosimulator(simulator)
# display(sim_serial)

# # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
# dump_pickled_dict(sim_serial, sim_serial_filepath)

###################################################################################################################
# START MODIFYING SCRIPT FROM HERE
###################################################################################################################

from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder

# Select the regions for the fine scale modeling with NetPyNE spiking networks
# numbers come from overall connectome, so find where CCTC regions are. Add more for e.g. DCN and NO if they're the same region? or just do thalamus for now
spiking_nodes_ids = [1] # the indices of fine scale regions modeled with NetPyNE

# Build a ANNarchy network model with the corresponding builder
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import load_netpyne
netpyne = load_netpyne(config=config)

spiking_network_builder = NetpyneNetworkBuilder(simulator, spiking_nodes_ids, spiking_simulator=netpyne, config=config)


# # ----------------------------------------------------------------------------------------------------------------
# # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# # ----------------------------------------------------------------------------------------------------------------
from copy import deepcopy

spiking_network_builder.population_order = 200 # reduce for speed

##TODO: add rest of cfg/netParams function, or is this enough?
from tvb_multiscale.tvb_netpyne.netpyne_models.models.thalamic_VIM_ET.src.netParams import netParams
from tvb_multiscale.tvb_netpyne.netpyne_models.models.thalamic_VIM_ET.src.cfg import cfg

# Populations' configurations
# When any of the properties model, params and scale below depends on regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property

# TODO: allow for having several nodes in single entry
vim_node_id = [1]
spiking_network_builder.populations = [
    {"label": "TC_pop", 
     "params":  {"global_label": "TC_pop"}, 
     "nodes": vim_node_id,  # Eth in [4]
     "scale": 1.0}
]

# Within region-node connections
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property

synapse_model = None
# conn_spec = {'rule': "all_to_all", 
#              "allow_self_connections": True, "force_multiple_weights": False}
# conn_spec_fixed_probability = conn_spec.copy()
# conn_spec_fixed_probability.update({'rule': "fixed_probability", "probability": 0.1})

def conn_spec_fixed_prob(prob=None):
    return {"rule": {"prob": prob}}

within_node_delay = 1.0  # ms

"""
# for each connection, we have a different probability
spiking_network_builder.populations_connections = [
     #        source   ->   target
    {"source": "I", "target": "I",  # I -> I This is a self-connection for population "Igpe"
     "synapse_model": synapse_model, "conn_spec": conn_spec_fixed_prob(pGPeGPe),  # conn_spec
     "weight": np.abs(wGPeGPe).item(), "delay": within_node_delay,
     "receptor_type": "gaba", "nodes": Igpe_nodes_ids},  # None means apply to all
    {"source": "I", "target": "I",  # I -> I This is a self-connection for population "Igpi"
     "synapse_model": synapse_model, "conn_spec": conn_spec_fixed_prob(pGPiGPi),  # conn_spec
     "weight": np.abs(wGPiGPi).item(), "delay": within_node_delay,
     "receptor_type": "gaba", "nodes": Igpi_nodes_ids},  # None means apply to all
    {"source": "IdSN", "target": "IdSN",  # IdSN -> IdSN This is a self-connection for population "IdSN"
     "synapse_model": synapse_model, "conn_spec": conn_spec_fixed_prob(pdSNdSN),  # conn_spec
     "weight": np.abs(wdSNdSN).item(), "delay": within_node_delay,
     "receptor_type": "gaba", "nodes": Istr_nodes_ids},
    {"source": "IiSN", "target": "IiSN",  # IiSN -> IiSN This is a self-connection for population "IiSN"
     "synapse_model": synapse_model, "conn_spec": conn_spec_fixed_prob(piSNiSN),  # conn_spec
     "weight": np.abs(wiSNiSN).item(), "delay": within_node_delay,
     "receptor_type": "gaba", "nodes": Istr_nodes_ids},
    ]
"""

# Among/Between region-node connections
# Given that only the AMPA population of one region-node couples to
# all populations of another region-node,
# we need only one connection type
        
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# depends on regions, set a handle to a function with
# arguments (source_region_index=None, target_region_index=None)

from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay

# We set global coupling scaling to 1.0,
# because we need the Maith et al optimized weights without any scaling:
spiking_network_builder.global_coupling_scaling = 1.0 
        
# Function that will return the TVB weight with optional scaling:
class TVBWeightFun(object):
    
    def __init__(self, 
                 global_coupling_scaling=spiking_network_builder.global_coupling_scaling, 
                 tvb_weights = spiking_network_builder.tvb_weights):
        self.global_coupling_scaling = float(global_coupling_scaling)
        self.tvb_weights = tvb_weights.copy()
    
    def __call__(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.global_coupling_scaling)

# Function that will return the TVB delay unless SPIKING_NODES_DELAYS == False:
tvb_delay_fun = \
    lambda source_node, target_node: \
        np.maximum(spiking_network_builder.tvb_dt, 
                   tvb_delay(source_node, target_node, spiking_network_builder.tvb_delays)) \
            if SPIKING_NODES_DELAYS else within_node_delay

# Creating  devices to be able to observe ANNarchy activity:

spiking_network_builder.output_devices = []

period = 1.0

# Creating  devices to be able to observe ANNarchy activity:
params = {} # deepcopy(spiking_network_builder.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["SpikeMonitor"])
for pop in spiking_network_builder.populations:
    connections = OrderedDict({})
    #                      label <- target population
    connections[pop["label"]] = pop["label"]
    spiking_network_builder.output_devices.append(
        {"model": "spike_recorder", "params": deepcopy(params),
         "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

# Labels have to be different for every connection to every distinct population
    
    
# # Create a spike stimulus input device
spiking_network_builder.input_devices = []  #

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# This will be transferred to NetPyNE
config.simulation_length = simulator.simulation_length

spiking_network_builder.configure(netParams=netParams, simConfig=cfg, autoCreateSpikingNodes=False)
# confirmed: now spiking_network_builder.global_coupling_scaling = 0.00390625


def synaptic_weight_scale_func(is_coupling_mode_tvb):
    # TODO: to be tuned or removed?
    if is_coupling_mode_tvb: # "TVB"
        return 1 # 1e-2
    else: # "spikeNet"
        return 1 # 5

# spiking_network_builder.global_coupling_scaling *= simulator.model.G
spiking_network_builder.netpyne_synaptic_weight_scale = synaptic_weight_scale_func(is_coupling_mode_tvb=INTERFACE_COUPLING_MODE=="TVB")


netpyne_network = spiking_network_builder.build() # set_defaults=False
# confirmed: 6 spiking pops of 200 neurs

for i, (connId, conn) in enumerate(spiking_network_builder.netpyne_instance.netParams.connParams.items()):
    print(f"{i}. {connId}: {conn.get('weight')} {conn.get('probability')}")
# confirmed: 13 connections between pops (weights and probs confirmed)

# confirmed: 6 spike recorders, as in ANNarchy when "_ts" stuff commented out

# TODO: 6 pops vs 4 in annarchy (but should be okay, because of different way of setting spiking_network_builder.populations)
populations_sizes = []
print("Population sizes: ")
for pop in spiking_network_builder.populations:
    populations_sizes.append(int(np.round(pop["scale"] * spiking_network_builder.population_order)))
    print("%s: %d" % (pop["label"], populations_sizes[-1]))

from tvb_multiscale.tvb_netpyne.interfaces.builders import TVBNetpyneInterfaceBuilder
    
# Build a TVB-ANNarchy interface with all the appropriate connections between the
# TVB and ANNarchy modelled regions
tvb_spikeNet_model_builder = TVBNetpyneInterfaceBuilder()

tvb_spikeNet_model_builder.config = config
tvb_spikeNet_model_builder.tvb_cosimulator = simulator            
tvb_spikeNet_model_builder.spiking_network = netpyne_network
# This can be used to set default tranformer and proxy models:
tvb_spikeNet_model_builder.model = "RATE"          # "RATE" (or "SPIKES", "CURRENT") TVB->ANNarchy interface
tvb_spikeNet_model_builder.input_flag = True   # If True, NetPyNE->TVB update will be implemented
tvb_spikeNet_model_builder.output_flag = True  # If True, TVB->NetPyNE coupling will be implemented
# If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
# and then applied with no time delay via a single "TVB proxy node" / ANNarchy device for each spiking region,
# "1-to-1" TVB->ANNarchy coupling.
# If any other value, we need 1 "TVB proxy node" / ANNarchy device for each TVB sender region node, and
# large-scale coupling for spiking regions is computed in ANNarchy, 
# taking into consideration the TVB connectome weights and delays, 
# in this "1-to-many" TVB->ANNarchy coupling.
tvb_spikeNet_model_builder.default_coupling_mode = INTERFACE_COUPLING_MODE # "spikeNet" # "TVB" 
# Number of neurons per population to be used to compute population mean instantaneous firing rates:
tvb_spikeNet_model_builder.proxy_inds = np.array(spiking_nodes_ids)
tvb_spikeNet_model_builder.N_E = spiking_network_builder.population_order
tvb_spikeNet_model_builder.TC_proxy_inds = np.array(vim_node_id)

# Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
tvb_spikeNet_model_builder.exclusive_nodes = True  

tvb_spikeNet_model_builder.output_interfaces = []
tvb_spikeNet_model_builder.input_interfaces = []

from tvb_multiscale.tvb_netpyne.interfaces.builders import NetpyneInputProxyModels

# TVB applies a global coupling scaling of coupling.a * model.G
tvb_spikeNet_model_builder.global_coupling_scaling = \
    tvb_spikeNet_model_builder.tvb_cosimulator.coupling.a[0].item() * simulator.model.G
print("global_coupling_scaling = %g" % tvb_spikeNet_model_builder.global_coupling_scaling)

#TVB (cortex) connects to & provides INPUT to spiking region TC_pop (vim)

# Total TVB indegree weight to TC_pop:
wTVBTCs = simulator.connectivity.weights[vim_node_id, 5:].squeeze()
print("simulator.connectivity.weights")
print(simulator.connectivity.weights[4,5:])
print("wTVBTCs")
print(wTVBTCs)
wTVBTC = wTVBTCs.sum().item()
print("wTVBTC = %g" % wTVBTC)
CTXtoTCinds = 5 + np.where(wTVBTC > 0.0)[0] # indices of TVB regions coupling to TC

# Approximate effective scaling of TVB coupling to TC_pop
# after normalizing with TC indegree and 
# multiplying with the Maith et al. optimized CTX -> TC weight
# TODO:this isn't in maith. should we still have this line here? if so, how do we set the value for this?
wCtxTC = 2 * 1e-3 # random temp value I picked in maith value range
pCtxTC = 0.5 # random temp value I picked (maith value range unknown)
iwCtxTC = TC_factor*wCtxTC / wTVBTC
print("iwCtxTC = %g" % iwCtxTC)

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------- BUILDER --------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

from tvb_multiscale.core.interfaces.base.transformers.models.red_wong_wang import RedWongWangExc


# --------For spike transmission from TVB to ANNarchy devices acting as TVB proxy nodes with TVB delays:--------


# TVB -> NetPyNE


if tvb_spikeNet_model_builder.default_coupling_mode == "spikeNet":
    
    # If coupling is computing in NetPyNE, we need as many TVB proxies 
    # as TVB regions coupling to Thalamus
    proxy_inds = np.unique(np.concatenate([CTXtoTCinds]))
    print("############## proxy_inds ################")
    print(proxy_inds)
    print(type(proxy_inds))
    # This is the TVB coupling weight function that will determine the connections' weights 
    # from TVB proxies to the target TC population:
    class TVBWeightFunInterface(object):
    
        def __init__(self, scaling):
            self.scaling = float(scaling)

        def __call__(self, source_node, target_node, tvb_weights):
            return (scale_tvb_weight(source_node, target_node, tvb_weights, scale=self.scaling))

    # A similar function for TVB coupling delays is also applied in the background 
    # without need to be explicitly defined by the user

tvb_spikeNet_model_builder.synaptic_weight_scale_func = synaptic_weight_scale_func
    
tvb_spikeNet_model_builder.output_interfaces = []
# Mean spike rates are applied in parallel to all target neurons
for trg_pop, target_nodes, conn_scaling, this_conn_spec, scale_factor in \
                  zip(["TC_pop"], # NetPyNE target populations
                      # Target region indices in NetPyNE:  
                      [tvb_spikeNet_model_builder.TC_proxy_inds], 
                      # Maith et al optimized... 
                      [wCtxTC], # ...weights 
                      # ...and probabilities for CTX -> TC_pop connections
                      [conn_spec_fixed_prob(prob=pCtxTC),  # pCtxTC  
                       ], 
                      # Interface scaling factors scaled by TVB weights' indegree to TC:
                      [TC_factor/wTVBTC]):          
        tvb_spikeNet_model_builder.output_interfaces.append(
            {"voi": np.array(["R"]),             # Source TVB state variable
             "populations": np.array([trg_pop]), # NetPyNE target population
             "model": "RATE",
             "spiking_proxy_inds": target_nodes, # Target region indices in NetPyNE
             # This spike generator device generates spike trains 
             # with autocorrelation corr at a time scale tau
             "proxy_model": NetpyneInputProxyModels.RATE,  # ANNarchyInputProxyModels.RATE_TO_SPIKES, # 
             
             # TODO: above needed?

             "proxy_params": {"geometry": 600, "record": ["spike"],
                              "corr": 0.3, "tau": 10.0, # comment for RATE_TO_SPIKES
                              "number_of_neurons": tvb_spikeNet_model_builder.N_E, # todo: de-hardcode
                              },
             'conn_spec': this_conn_spec,  # dictionary of connection properties
             'coupling_mode': tvb_spikeNet_model_builder.default_coupling_mode
        })  # None means all here
        
        # For both coupling modes, we scale the TVB rate already at the TVB -> ANNarchy transformer
        # with the interface scale factor (normalized by TVB indegree to STN/Striatum)
        # and the global coupling scaling.
        if tvb_spikeNet_model_builder.output_interfaces[-1]["coupling_mode"] == "spikeNet":
            tvb_spikeNet_model_builder.output_interfaces[-1]["proxy_inds"] = proxy_inds
            # In this case connections from each TVB proxy to TC 
            # are scaled additionally with the Maith et al. optimized weights
            tvb_spikeNet_model_builder.output_interfaces[-1]["weights"] = TVBWeightFunInterface(conn_scaling)
            tvb_spikeNet_model_builder.output_interfaces[-1]["transformer_params"] = \
                {"scale_factor": scale_factor * tvb_spikeNet_model_builder.global_coupling_scaling}
            # In total:
            # From each TVB proxy node we get a rate scaled as (
            # (coupling.a * G * STN_factor/wTVBSTN) * R_i, (i for all TVB regions)
            # Then, spikes generated from each TVB proxy are transferred via connections 
            # with weights TVB_w_ji * wCtxSTN or wCtxiSN or wCtxdSN (j for STN or Striatum) 
            # and probabilities pCtxSTN or pCtxiSN or pCtxdSN, respectively
        else:
            # In this case connections from each TVB proxy to TC_pop 
            # are equal to the Maith et al. optimized weights
            tvb_spikeNet_model_builder.output_interfaces[-1]["weights"] = conn_scaling
            # In this case coupling.a is already applied during computing TVB coupling.
            # Therefore we scale only with model.G
            tvb_spikeNet_model_builder.output_interfaces[-1]["transformer_params"] = \
                {"scale_factor": scale_factor * tvb_spikeNet_model_builder.G}
            # In total:
            # From each TVB proxy node we get a total coupling rate scaled 
            # as (coupling.a * G STN_factor/wTVBSTN) * R_j, (j for STN or Striatum)
            # Then, spikes generated from each TVB proxy are transferred via connections 
            # with weights wCtxSTN or wCtxiSN or wCtxdSN and 
            # probabilities pCtxSTN or pCtxiSN or pCtxdSN, respectively
            
            
from tvb_multiscale.core.interfaces.base.transformers.models.red_wong_wang import ElephantSpikesRateRedWongWangExc

tvb_spikeNet_model_builder.input_interfaces = []
# TVB <-- ANNarchy:
for src_pop, nodes, in zip(
                # Source populations in NetPyNE:
                [np.array(["TC_pop"])],
                # Source regions indices in NetPyNE:
                [vim_node_id]):          # Thalamus
        #            TVB <- NetPyNE
        tvb_spikeNet_model_builder.input_interfaces.append(
            {"voi": np.array(["S", "R"]),  # Target state variables in TVB
             "populations": src_pop,  # Source populations in NetPyNE
             # This transformer not only converts spike counts to rates for state variable R,
             # but also integrates the dS/dt to compute the respective S!:
             "transformer": ElephantSpikesRateRedWongWangExc,
             "transformer_params": 
                 # Spike counts are converted to rates via:
                 # number_of_spikes / number_of_neurons_per_population / number_of_populations
                 # (mind that there are 2 populations in Striatum)
                 {"scale_factor": np.array([1.0]) / tvb_spikeNet_model_builder.N_E / len(src_pop),
                  # The integrator used to integrate dS/dt
                   "integrator":CONFIGURED.DEFAULT_TRANSFORMER_INTEGRATOR_MODEL(
                                       dt=simulator.integrator.dt),
                    "state": np.zeros((2, len(nodes))), # initial condition
                    # Parameters of the dS/dt differential equation:
                    "tau_s": simulator.model.tau_s, # time constant of integrating S
                    "tau_r": np.array([10.0]),      # time constant of integrating R to low pass filter it
                    "gamma": simulator.model.gamma}, 
             "proxy_inds": np.array(nodes)})  # None means all here
    

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Configure and build:
tvb_spikeNet_model_builder.configure()
tvb_spikeNet_model_builder.print_summary_info_details(recursive=1)
    
print("\noutput (TVB->NetPyNE coupling) interfaces' configurations:\n")
display(tvb_spikeNet_model_builder.output_interfaces)
    
print("\ninput (NetPyNE->TVB update) interfaces' configurations:\n")
display(tvb_spikeNet_model_builder.input_interfaces)

print(proxy_inds)

simulator = tvb_spikeNet_model_builder.build()

# NetPyNE model is built in two steps. First need to create declarative-style specification for both spiking network itself and TVB-Netpyne proxy devides (interfaces).
# Once it's done above using builders, network can be instantiated based on the specification:

netpyne.simConfig.duration = simulation_length # TODO: do it properly! (find out why it gets lost)
netpyne.simConfig.dt = spiking_network_builder.spiking_dt
netpyne.allowSelfConns = True

netpyne_network.netpyne_instance.instantiateNetwork() #netpyne structure created
# can inspect connectivity params

simulator.simulate_spiking_simulator = netpyne_network.netpyne_instance.run  # set the method to run NetPyNE
    
# simulator.print_summary_info(recursive=3)
# simulator.print_summary_info_details(recursive=3)
    
print("\n\noutput (TVB->NetPyNE coupling) interfaces:\n")
simulator.output_interfaces.print_summary_info_details(recursive=2)
    
print("\n\ninput (NetPyNE->TVB update) interfaces:\n")
simulator.input_interfaces.print_summary_info_details(recursive=2)

# -----------------------------------4. Compile network ---------------------------------------------------------
# Compile the ANNarchy network...
# tic_compile = time.time()
# netpyne_network.configure()
# print("Compiled! in %f min" % ((time.time() - tic_compile) / 60))

netpyne_network.print_summary_info_details(recursive=2, connectivity=False)

 # -----------------------------------5. Simulate and gather results-------------------------------------------------
# simulation_length = 1500.0
# transient = 500.0 # simulation_length/11
# ...and simulate!

simulator.configure() # 

simulation_length = \
            np.ceil(simulation_length / simulator.synchronization_time) * simulator.synchronization_time

advance_simulation_for_delayed_monitors_output = True
if simulation_mode == "rs":
    simulation_length1 = simulation_length
else:
    start_stimulus = np.ceil(start_stimulus / simulator.synchronization_time) * simulator.synchronization_time
    simulation_length1 = start_stimulus
    advance_simulation_for_delayed_monitors_output = False

t_start = time.time()

results = simulator.run(simulation_length=simulation_length1, 
                       advance_simulation_for_delayed_monitors_output=advance_simulation_for_delayed_monitors_output
                       ) # 35.0 with stimulus application
    
print("\nSimulated in %f secs!" % (time.time() - t_start))

netpyne.finalize()

#from netpyne import sim
#popIds = [id for id in sim.net.pops.keys()][4] # only thalamus
#sa = sim.analysis
#%matplotlib inline
# sim.analysis.plotConn(showFig=True, includePre=popIds, includePost=popIds, feature='weight');
# sim.analysis.plotConn(showFig=True, includePre=popIds, includePost=popIds, feature='prob');
# sim.analysis.plotConn(showFig=True, includePre=popIds, includePost=popIds, feature='probability');
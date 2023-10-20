# Import
from __future__ import print_function
from inspect import trace  # python 2 & 3 compatible

import time
from tkinter.tix import CELL
import h5py
import random
from operator import itemgetter
from multiprocessing import cpu_count

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import nest
import numpy as np
import os

from pynestml.frontend.pynestml_frontend import generate_nest_target

#NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("/home/neurocomp/workspace/nest-simulator-install")


 #generate_nest_target(input_path="/home/neurocomp/workspace/eglif/eglif_cond_alpha_multisyn.nestml")
                     
                     
nest.Install("cerebmodule")

eglif_neu= nest.Create("eglif_cond_alpha_multisyn")

print(nest.GetDefaults("eglif_cond_alpha_multisyn"))



# Load file with positions and connections data
hdf5_file = 'balanced_DCN_IO.hdf5'


###### PARAMETERS SETTING ######################################################
# Synapse parameters: in E-GLIF, 3 synaptic receptors are present: the first is always associated to exc, the second to inh, the third to remaining synapse type
Erev_exc = 0.0		# [mV]	#[Cavallari et al, 2014]
Erev_inh = -80.0		# [mV]
tau_exc = {'golgi': 0.23, 'granule': 5.8, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0, 'dcnp': 3.64, 'io': 1.0}		#tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
tau_inh = {'golgi': 10.0, 'granule': 13.61, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7, 'dcnp': 1.14, 'io': 60.0}
tau_exc_cfpc = 0.4
tau_exc_pfgoc = 0.5
tau_exc_cfmli = 1.2

# Single neuron parameters:
neuron_param = {'golgi_cell': {'t_ref': 2.0, 'C_m': 145.0,'tau_m': 44.0,'V_th': -55.0,'V_reset': -75.0,'Vinit': -62.0,'E_L': -62.0,'V_min':-150.0,
                         'lambda_0':1.0, 'tau_V':0.4,'I_e': 16.214,'kadap': 0.217,'k1': 0.031, 'k2': 0.023,'A1': 259.988,'A2':178.01,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['golgi'], 'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
               'granule_cell': {'t_ref': 1.5, 'C_m': 7.0,'tau_m': 24.15,'V_th': -41.0,'V_reset': -70.0,'Vinit': -62.0,'E_L': -62.0,'V_min': -150.0,
                           'lambda_0':1.0, 'tau_V':0.3,'I_e': -0.888,'kadap': 0.022,'k1': 0.311, 'k2': 0.041,'A1': 0.01,'A2':-0.94,
                           'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['granule'], 'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
               'purkinje_cell': {'t_ref': 0.5, 'C_m': 334.0,'tau_m': 47.0,'V_th': -43.0,'V_reset': -69.0,'Vinit': -59.0,'E_L': -59.0,
                            'lambda_0':4.0, 'tau_V':3.5,'I_e': 176.26,'kadap': 1.492,'k1': 0.1950, 'k2': 0.041,'A1': 157.622,'A2':172.622,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['purkinje'], 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
               'basket_cell': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                          'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
               'stellate_cell': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                            'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
               'dcn_cell_glut_large': {'t_ref': 1.5, 'C_m': 142.0,'tau_m': 33.0,'V_th': -36.0,'V_reset': -55.0,'Vinit': -45.0,'E_L': -45.0,
                            'lambda_0': 3.5, 'tau_V': 3.0,'I_e': 75.385,'kadap': 0.408,'k1': 0.697, 'k2': 0.047,'A1': 13.857,'A2':3.477,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcn'], 'tau_syn2': tau_inh['dcn']},
               'dcn_cell_GABA': {'t_ref': 3.0, 'C_m': 56.0,'tau_m': 56.0,'V_th': -39.0,'V_reset': -55.0,'Vinit': -40.0,'E_L': -40.0,
                            'lambda_0': 0.9, 'tau_V': 1.0,'I_e': 2.384,'kadap': 0.079,'k1': 0.041, 'k2': 0.044,'A1': 176.358,'A2':176.358,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcnp'], 'tau_syn2': tau_inh['dcnp']},
               'io_cell': {'t_ref': 1.0, 'C_m': 189.0,'tau_m': 11.0,'V_th': -35.0,'V_reset': -45.0,'Vinit': -45.0,'E_L': -45.0,
               	     'lambda_0':1.2, 'tau_V': 0.8,'I_e': -18.01,'kadap': 1.928,'k1': 0.191, 'k2': 0.091,'A1': 1810.923,'A2': 1358.197,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['io'], 'tau_syn2': tau_inh['io']}}



# Connection weights
conn_weights = {'mossy_to_glomerulus': 1.0,'ascending_axon_to_golgi': 0.822, 'ascending_axon_to_purkinje': 0.882, 'basket_to_purkinje': 0.436, 'basket_to_basket': 0.006, \
                'glomerulus_to_golgi': 0.240, 'glomerulus_to_granule':  0.232, 'golgi_to_granule': 0.148, 'golgi_to_golgi': 0.00696, \
                'parallel_fiber_to_basket': 0.1, 'parallel_fiber_to_golgi': 0.054,'parallel_fiber_to_purkinje': 0.136,\
                'parallel_fiber_to_stellate': 0.178, 'stellate_to_purkinje': 1.642, 'stellate_to_stellate': 0.005,\
                'purkinje_to_dcn_glut_large': 0.297, 'mossy_to_dcn_glut_large': 0.554, 'purkinje_to_dcn_GABA': 0.072,\
                'io_to_purkinje': 300.0, 'io_to_basket': 3.0, 'io_to_stellate': 11.0, 'io_to_dcn_glut_large': 1.5, 'io_to_dcn_GABA': 0.3,'dcn_GABA_to_io':0.004}


# Connection delays
conn_delays = {'mossy_to_glomerulus': 1.0,'ascending_axon_to_golgi': 2.0, 'ascending_axon_to_purkinje': 2.0, 'basket_to_purkinje': 4.0, 'basket_to_basket': 4.0, \
               'glomerulus_to_golgi': 1.0, 'glomerulus_to_granule': 1.0, 'golgi_to_granule': 2.0, 'golgi_to_golgi': 4.0,\
               'parallel_fiber_to_basket': 5.0, 'parallel_fiber_to_golgi': 5.0,'parallel_fiber_to_purkinje': 5.0,
               'parallel_fiber_to_stellate': 5.0, 'stellate_to_purkinje':5.0, 'stellate_to_stellate': 4.0, \
               'purkinje_to_dcn_glut_large': 4.0, 'mossy_to_dcn_glut_large': 4.0, 'purkinje_to_dcn_GABA': 4.0,\
                'io_to_purkinje': 4.0, 'io_to_basket': 80.0, 'io_to_stellate': 80.0, 'io_to_dcn_glut_large': 4.0, 'io_to_dcn_GABA': 5.0, 'dcn_GABA_to_io': 25.0}

# Connection receptors
conn_receptors = {'ascending_axon_to_golgi': 3, 'ascending_axon_to_purkinje': 1, 'basket_to_purkinje': 2,
               'glomerulus_to_golgi': 1, 'glomerulus_to_granule': 1, 'golgi_to_granule': 2, 'golgi_to_golgi': 2,
               'parallel_fiber_to_basket': 1, 'parallel_fiber_to_golgi': 3,'parallel_fiber_to_purkinje': 1,
               'parallel_fiber_to_stellate': 1, 'stellate_to_purkinje': 2, 'stellate_to_stellate': 2,
               'basket_to_basket': 2, 'purkinje_to_dcn_glut_large': 2, 'mossy_to_dcn_glut_large': 1, 'purkinje_to_dcn_GABA': 2,\
               'io_to_purkinje': 3, 'io_to_basket': 3, 'io_to_stellate': 3, 'io_to_dcn_glut_large': 1, 'io_to_dcn_GABA': 1, 'dcn_GABA_to_io': 2}


# Connection pre and post-synaptic neurons
conn_pre_post = {'mossy_to_glomerulus': {'pre': 'mossy_fibers', 'post': 'glomerulus'},\
                 'ascending_axon_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'},\
                 'ascending_axon_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'},\
                 'basket_to_purkinje': {'pre': 'basket_cell', 'post': 'purkinje_cell'},\
                 'glomerulus_to_golgi': {'pre': 'glomerulus', 'post': 'golgi_cell'}, \
                 'glomerulus_to_granule': {'pre': 'glomerulus', 'post': 'granule_cell'}, \
                 'golgi_to_granule': {'pre': 'golgi_cell', 'post': 'granule_cell'},\
                 'golgi_to_golgi': {'pre': 'golgi_cell', 'post': 'golgi_cell'},\
                 'parallel_fiber_to_basket': {'pre': 'granule_cell', 'post': 'basket_cell'}, \
                 'parallel_fiber_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'},\
                 'parallel_fiber_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'},\
                 'parallel_fiber_to_stellate': {'pre': 'granule_cell', 'post': 'stellate_cell'}, \
                 'stellate_to_purkinje': {'pre': 'stellate_cell', 'post': 'purkinje_cell'}, \
                 'basket_to_basket': {'pre': 'basket_cell', 'post': 'basket_cell'},\
                 'stellate_to_stellate': {'pre': 'stellate_cell', 'post': 'stellate_cell'}, \
                 'mossy_to_dcn_glut_large': {'pre': 'mossy_fibers', 'post': 'dcn_cell_glut_large'}, \
                 'purkinje_to_dcn_glut_large': {'pre': 'purkinje_cell', 'post': 'dcn_cell_glut_large'}, \
                 'purkinje_to_dcn_GABA': {'pre': 'purkinje_cell', 'post': 'dcn_cell_GABA'},\
                 'io_to_purkinje': {'pre': 'io_cell', 'post': 'purkinje_cell'}, \
                 'io_to_basket': {'pre': 'io_cell', 'post': 'basket_cell'}, \
                 'io_to_stellate': {'pre': 'io_cell', 'post': 'stellate_cell'}, \
                 'io_to_dcn_glut_large': {'pre': 'io_cell', 'post': 'dcn_cell_glut_large'},
                 'io_to_dcn_GABA': {'pre': 'io_cell', 'post': 'dcn_cell_GABA'},
                 'dcn_GABA_to_io': {'pre': 'dcn_cell_GABA', 'post': 'io_cell'}}

high_iomli = 120.0          # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
min_iomli = 40.0

######################## NEST simulation setup ##########################################
nest.ResetKernel()
nest.set_verbosity('M_ERROR')
nest.SetKernelStatus({"overwrite_files": True, "data_path": "sim_data/"})


f = h5py.File(hdf5_file, 'r+')

neuron_types = list(f['cells/placement'].keys())
print(neuron_types)

neuron_number = {}
start_id_scaffold = {}

# Create a dictionary; keys = cell names, values = lists to store neuron models
neuron_models = {key: [] for key in neuron_types}

# All cells are modelled as E-GLIF models;
# with the only exception of Glomeruli and Mossy Fibers (not cells, just modeled as
# relays; i.e., parrot neurons)
neuron_types.remove('dcn_cell_Gly-I')
for neuron_name in neuron_types:
    if neuron_name  != 'glomerulus' and neuron_name != 'mossy_fibers':
        if neuron_name not in nest.Models():
            nest.CopyModel('eglif_cond_alpha_multisyn', neuron_name)
            nest.SetDefaults(neuron_name, neuron_param[neuron_name])
    else:
        if neuron_name not in nest.Models():
            nest.CopyModel('parrot_neuron', neuron_name)

    neuron_number[neuron_name] = np.array(f['cells/placement/'+neuron_name+'/identifiers'])[1]
    start_id_scaffold[neuron_name] = np.array(f['cells/placement/'+neuron_name+'/identifiers'])[0]

    neuron_models[neuron_name] = nest.Create(neuron_name, neuron_number[neuron_name])



### Load connections from hdf5 file and create them in NEST:
for conn in conn_weights.keys():
    exec("conn_name = conn")
    exec("conn = np.array(f['cells/connections/'+conn_name])")
    exec('pre_name = conn_pre_post[conn_name]["pre"]')
    exec('post_name = conn_pre_post[conn_name]["post"]')
    exec("pre = np.array(conn[:,0])-start_id_scaffold[pre_name]+neuron_models[pre_name][0]")
    exec("post = np.array(conn[:,1])-start_id_scaffold[post_name]+neuron_models[post_name][0]")
    pre=list(pre.astype(int))
    post=list(post.astype(int))
    print("Connecting  ", conn_name, "!")

    if conn_name=="mossy_to_glomerulus":
        syn_param = {"synapse_model": "static_synapse", "weight": np.ones(len(pre))*[conn_weights[conn_name]], "delay": np.ones(len(pre))*conn_delays[conn_name]}
    elif conn_name == "io_bc" or conn_name == "io_sc":
        syn_param = {"synapse_model": "static_synapse", "weight": np.ones(len(pre))*conn_weights[conn_name], \
                     "delay": {'distribution': 'exponential_clipped_to_boundary', 'low': min_iomli, 'high': high_iomli, 'lambda': conn_delays[conn]},"receptor_type":conn_receptors[conn_name]}
    # elif conn_name == "dcn_GABA_to_io":
    #     print([conn_delays], conn_name)
    #     syn_param = {"synapse_model": "static_synapse", \
    #         "weight": np.ones(len(pre))*[conn_weights[conn_name]],    #nest.random.normal(mean=conn_delays[conn_name], std=conn_delays[conn_name]/10),\
    #     # {'distribution': 'exponential_clipped_to_boundary', 'low': min_iomli, \
    #     #     'high': high_iomli, \
    #     #         'lambda': list(np.ones(len(pre))*[conn_delays[conn_name]])}, \
    #                  "delay": np.ones(len(pre))*[conn_delays[conn_name]],"receptor_type":conn_receptors[conn_name]}
    else:
        syn_param = {"synapse_model": "static_synapse", "weight": np.ones(len(pre))*[conn_weights[conn_name]], "delay": np.ones(len(pre))*conn_delays[conn_name],"receptor_type":conn_receptors[conn_name]}
    
    nest.Connect(pre, post, {"rule": "one_to_one"}, syn_param)


###################### NEST simulation #########################################
RECORD_VM = False
TOT_DURATION = 2000. # mseconds
STIM_START = 100. # beginning of stimulation
STIM_END = 200.   # end of stimulation
BURST_FREQ = 100.  # Frequency in Hz
BACKGROUND_FREQ = 4.
STIM_RATE = 10.




# Background as Poisson process
background = nest.Create('poisson_generator',params={'rate':BACKGROUND_FREQ, 'start': 0.0, 'stop': TOT_DURATION})
nest.Connect(background,neuron_models['mossy_fibers'])


# Sinusoidal whisking stimulus
whisking_stim = nest.Create('sinusoidal_poisson_generator', params={
            "rate": STIM_RATE,
            "amplitude": 15.0,
            "frequency": 10.0,
            "phase": 0.0
          })

# Localized CS
r_x, r_z = 75, 50
gloms_pos = np.array(f['cells/placement/glomerulus/positions'])
x_s_c, x_p_c, z_c = 75., 225., 100.


# Find glomeruli falling into the selected volume
target_gloms_bool = np.add(((gloms_pos[:,[0]] - x_s_c)**2)/r_x**2,((gloms_pos[:,[2]] - z_c)**2)/r_z**2).__lt__(1)              # ellipse equation
target_gloms_id_scaffold_spinal = np.array(np.where(target_gloms_bool)[0] + start_id_scaffold['glomerulus'])

target_gloms_bool = np.add(((gloms_pos[:,[0]] - x_p_c)**2)/r_x**2,((gloms_pos[:,[2]] - z_c)**2)/r_z**2).__lt__(1)              # ellipse equation
target_gloms_id_scaffold_principal = np.array(np.where(target_gloms_bool)[0] + start_id_scaffold['glomerulus'])

# Select the corrisponding original MFs
conn_glom_mf = np.array(f['cells/connections/mossy_to_glomerulus'])
target_mfs_id_scaffold_spinal = conn_glom_mf[np.isin(conn_glom_mf[:, 1],target_gloms_id_scaffold_spinal), 0]
target_mfs_id_scaffold_principal = conn_glom_mf[np.isin(conn_glom_mf[:, 1],target_gloms_id_scaffold_principal), 0]
# translate to NEST ids
target_mfs_id_nest_spinal = target_mfs_id_scaffold_spinal - start_id_scaffold['mossy_fibers'] + neuron_models['mossy_fibers'][0]
target_mfs_id_nest_spinal = target_mfs_id_nest_spinal.astype(int)
target_mfs_id_nest_principal = target_mfs_id_scaffold_principal - start_id_scaffold['mossy_fibers'] + neuron_models['mossy_fibers'][0]
target_mfs_id_nest_principal = target_mfs_id_nest_principal.astype(int)

# Obtain an ordered list of non-duplicates
id_stim_spinal = sorted(list(set(target_mfs_id_nest_spinal)))
id_stim_principal = sorted(list(set(target_mfs_id_nest_principal)))

# n = len(id_stim)
# print(n, " stimulated mfs")
# nest.Connect(list(CS[:n]), id_stim, {'rule': 'one_to_one'})

nest.Connect(whisking_stim,id_stim_spinal)
nest.Connect(whisking_stim,id_stim_principal)

## Record spikes from granule and Golgi cells
grc_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "granule_spikes"})
goc_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "golgi_spikes"})
glom_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "glomerulus_spikes"})
pc_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "purkinje_spikes"})
bc_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "basket_spikes"})
sc_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "stellate_spikes"})
dcn_spikes = nest.Create("spike_recorder",
                         params={"record_to": "memory", "label": "dcn_glut_large_spikes"})

# Here you can choose which devices you want to connect and thus the neural populations you want to record.
# Increasing the number of recorded cells can increase the duration of the simulation
nest.Connect(neuron_models['granule_cell'], grc_spikes)
nest.Connect(neuron_models['golgi_cell'], goc_spikes)
nest.Connect(neuron_models['glomerulus'], glom_spikes)
nest.Connect(neuron_models['purkinje_cell'], pc_spikes)
nest.Connect(neuron_models['basket_cell'], bc_spikes)
nest.Connect(neuron_models['stellate_cell'], sc_spikes)
nest.Connect(neuron_models['dcn_cell_glut_large'], dcn_spikes)




# RUN the simulation and collect spikes
nest.Simulate(TOT_DURATION)

# Get spike events for each recording device - the neural populations that you decided not to record will have empty spike detectors
goc_evs = goc_spikes.get()['events']['senders']
goc_times = goc_spikes.get()['events']['times']

grc_evs = grc_spikes.get()['events']['senders']
grc_evs = random.sample(list(grc_evs), int(len(grc_evs)*0.1))
grc_times = grc_spikes.get()['events']['times']
grc_times = random.sample(list(grc_times), int(len(grc_evs)*0.1))

glom_evs = glom_spikes.get()['events']['senders']
glom_times = glom_spikes.get()['events']['times']


pc_evs = pc_spikes.get()['events']['senders']
pc_times = pc_spikes.get()['events']['times']


sc_evs = sc_spikes.get()['events']['senders']
sc_times = sc_spikes.get()['events']['times']

bc_evs = bc_spikes.get()['events']['senders']
bc_times = bc_spikes.get()['events']['times']

dcn_evs = dcn_spikes.get()['events']['senders']
dcn_times = dcn_spikes.get()['events']['times']


# ######################### PLOTTING PSTH AND RASTER PLOTS ########################

CELL_TO_PLOT = ['glomerulus','granule_cell','basket_cell','stellate_cell','purkinje_cell','dcn_cell_glut_large']


cells = {'granule_cell': [grc_times, grc_evs],
         'golgi_cell': [goc_times, goc_evs],
         'glomerulus': [glom_times, glom_evs],
         'purkinje_cell': [pc_times, pc_evs],
         'stellate_cell': [sc_times, sc_evs],
         'basket_cell': [bc_times, bc_evs],
         'dcn_cell_glut_large': [dcn_times, dcn_evs]}


color = {'granule_cell': '#E62214',    #'rgba(255, 0, 0, .8)',
         'golgi_cell': '#332EBC',  #'rgba(0, 255, 0, .8)',
         'glomerulus': '#0E1030',             #rgba(0, 0, 0, .8)',
         'purkinje_cell': '#0F8944',   #'rgba(64, 224, 208, .8)',
         'stellate_cell': '#FFC425',       #'rgba(234, 10, 142, .8)',
         'basket_cell': '#F37735',
         'dcn_cell_glut_large': '#080808'}         #'rgba(234, 10, 142, .8)'}
# PSTH

def metrics(spikeData, TrialDuration, cell, figure_handle, sel_row):
    id_spikes = np.sort(np.unique(spikeData,return_index=True))
    bin_size = 5   # [ms]
    n_bins = int(TrialDuration/bin_size) + 1
    psth, tms = np.histogram(spikeData, bins = n_bins, range = (0 , TrialDuration))

    # absolute frequency
    abs_freq = np.zeros(id_spikes[0].shape[0])
    for idx,i in enumerate(id_spikes[0]):
        count = np.where(spikeData==i)[0]
        abs_freq[idx] = count.shape[0]

    # mean frequency
    m_f = (id_spikes[0].shape[0])/((TrialDuration/1000)*len(neuron_models[cell]))

   

    layout = go.Layout(
            scene = dict(aspectmode='data'),
            xaxis={'title':'time (ms)'},
            yaxis={'title':'number of spikes'}
    )

    figure_handle.add_trace(go.Bar(
        x = tms[0:len(tms)-1],
        y = psth/((bin_size*0.001)*neuron_number[cell]),
        width = 4.0,
        marker=dict(
            color=color[cell])
        ), row=sel_row, col=1)

    
  
  


    print("mean frequency: ", int(m_f))

    return tms

# RASTER
def raster(times, cell_ids, cell, fig_handle, sel_row):
    trace0 = go.Scatter(
    x = times,
    y = cell_ids,
    name = '',
    mode = 'markers',
    marker = dict(
        size = 4,
        color = color[cell],
        line = dict(
            width = .2,
            color = 'rgb(0, 0, 0)'
            )
        )
    )
    fig_handle.add_trace(trace0, row= sel_row, col=1)


fig_psth = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]', y_title='Frequency [Hz]')
fig_raster = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]', y_title='# cells')
num=1
for c in CELL_TO_PLOT:
    times = cells[c][0]
    cell_ids = cells[c][1]
    metrics(times, TOT_DURATION, c, fig_psth, num)
    raster(times, cell_ids, c, fig_raster, num)
    num+=1
fig_psth.update_xaxes(range=[1000,2000])
fig_raster.update_xaxes(range=[1000,2000])
fig_psth.update_layout(showlegend=False)
fig_raster.update_layout(showlegend=False)
fig_psth.show()
fig_raster.show()
fig_psth.write_image("images/snn_psth_whisking.svg")
fig_raster.write_image("images/snn_raster_whisking.svg")





 
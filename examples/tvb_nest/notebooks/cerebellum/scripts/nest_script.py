# -*- coding: utf-8 -*-

import h5py
from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_script import *


###### PARAMETERS SETTING ######################################################
# Synapse parameters: in E-GLIF, 3 synaptic receptors are present: the first is always associated to exc, the second to inh, the third to remaining synapse type
Erev_exc = 0.0  # [mV]	#[Cavallari et al, 2014]
Erev_inh = -80.0  # [mV]
tau_exc = {'golgi': 0.23, 'granule': 5.8, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0,
           'dcnp': 3.64,
           'io': 1.0}  # tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
tau_inh = {'golgi': 10.0, 'granule': 13.61, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7,
           'dcnp': 1.14, 'io': 60.0}
tau_exc_cfpc = 0.4
tau_exc_pfgoc = 0.5
tau_exc_cfmli = 1.2

# Single neuron parameters:
neuron_param = {
    'golgi_cell': {'t_ref': 2.0, 'C_m': 145.0, 'tau_m': 44.0, 'V_th': -55.0, 'V_reset': -75.0, 'Vinit': -62.0,
                   'E_L': -62.0, 'V_min': -150.0,
                   'lambda_0': 1.0, 'tau_V': 0.4, 'I_e': 16.214, 'kadap': 0.217, 'k1': 0.031, 'k2': 0.023,
                   'A1': 259.988, 'A2': 178.01,
                   'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['golgi'],
                   'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
    'granule_cell': {'t_ref': 1.5, 'C_m': 7.0, 'tau_m': 24.15, 'V_th': -41.0, 'V_reset': -70.0, 'Vinit': -62.0,
                     'E_L': -62.0, 'V_min': -150.0,
                     'lambda_0': 1.0, 'tau_V': 0.3, 'I_e': -0.888, 'kadap': 0.022, 'k1': 0.311, 'k2': 0.041,
                     'A1': 0.01, 'A2': -0.94,
                     'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['granule'],
                     'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
    'purkinje_cell': {'t_ref': 0.5, 'C_m': 334.0, 'tau_m': 47.0, 'V_th': -43.0, 'V_reset': -69.0, 'Vinit': -59.0,
                      'E_L': -59.0,
                      'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 176.26, 'kadap': 1.492, 'k1': 0.1950, 'k2': 0.041,
                      'A1': 157.622, 'A2': 172.622,
                      'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['purkinje'],
                      'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
    'basket_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                    'E_L': -68.0,
                    'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                    'A1': 5.953, 'A2': 5.863,
                    'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                    'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
    'stellate_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                      'E_L': -68.0,
                      'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                      'A1': 5.953, 'A2': 5.863,
                      'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                      'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
    'dcn_cell_glut_large': {'t_ref': 1.5, 'C_m': 142.0, 'tau_m': 33.0, 'V_th': -36.0, 'V_reset': -55.0,
                            'Vinit': -45.0, 'E_L': -45.0,
                            'lambda_0': 3.5, 'tau_V': 3.0, 'I_e': 75.385, 'kadap': 0.408, 'k1': 0.697, 'k2': 0.047,
                            'A1': 13.857, 'A2': 3.477,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcn'],
                            'tau_syn2': tau_inh['dcn']},
    'dcn_cell_GABA': {'t_ref': 3.0, 'C_m': 56.0, 'tau_m': 56.0, 'V_th': -39.0, 'V_reset': -55.0, 'Vinit': -40.0,
                      'E_L': -40.0,
                      'lambda_0': 0.9, 'tau_V': 1.0, 'I_e': 2.384, 'kadap': 0.079, 'k1': 0.041, 'k2': 0.044,
                      'A1': 176.358, 'A2': 176.358,
                      'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcnp'],
                      'tau_syn2': tau_inh['dcnp']},
    'io_cell': {'t_ref': 1.0, 'C_m': 189.0, 'tau_m': 11.0, 'V_th': -35.0, 'V_reset': -45.0, 'Vinit': -45.0,
                'E_L': -45.0,
                'lambda_0': 1.2, 'tau_V': 0.8, 'I_e': -18.01, 'kadap': 1.928, 'k1': 0.191, 'k2': 0.091,
                'A1': 1810.923, 'A2': 1358.197,
                'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['io'],
                'tau_syn2': tau_inh['io']}}

# Connection weights
conn_weights = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 0.822, 'ascending_axon_to_purkinje': 0.882,
                'basket_to_purkinje': 0.436, 'basket_to_basket': 0.006, \
                'glomerulus_to_golgi': 0.240, 'glomerulus_to_granule': 0.232, 'golgi_to_granule': 0.148,
                'golgi_to_golgi': 0.00696, \
                'parallel_fiber_to_basket': 0.1, 'parallel_fiber_to_golgi': 0.054,
                'parallel_fiber_to_purkinje': 0.136, \
                'parallel_fiber_to_stellate': 0.178, 'stellate_to_purkinje': 1.642, 'stellate_to_stellate': 0.005, \
                'purkinje_to_dcn_glut_large': 0.297, 'mossy_to_dcn_glut_large': 0.554,
                'purkinje_to_dcn_GABA': 0.072, \
                'io_to_purkinje': 300.0, 'io_to_basket': 3.0, 'io_to_stellate': 11.0, 'io_to_dcn_glut_large': 1.5,
                'io_to_dcn_GABA': 0.3, 'dcn_GABA_to_io': 0.004}

# Connection delays
conn_delays = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 2.0, 'ascending_axon_to_purkinje': 2.0,
               'basket_to_purkinje': 4.0, 'basket_to_basket': 4.0, \
               'glomerulus_to_golgi': 1.0, 'glomerulus_to_granule': 1.0, 'golgi_to_granule': 2.0,
               'golgi_to_golgi': 4.0, \
               'parallel_fiber_to_basket': 5.0, 'parallel_fiber_to_golgi': 5.0, 'parallel_fiber_to_purkinje': 5.0,
               'parallel_fiber_to_stellate': 5.0, 'stellate_to_purkinje': 5.0, 'stellate_to_stellate': 4.0, \
               'purkinje_to_dcn_glut_large': 4.0, 'mossy_to_dcn_glut_large': 4.0, 'purkinje_to_dcn_GABA': 4.0, \
               'io_to_purkinje': 4.0, 'io_to_basket': 80.0, 'io_to_stellate': 80.0, 'io_to_dcn_glut_large': 4.0,
               'io_to_dcn_GABA': 5.0, 'dcn_GABA_to_io': 25.0}

# Connection receptors
conn_receptors = {'ascending_axon_to_golgi': 3, 'ascending_axon_to_purkinje': 1, 'basket_to_purkinje': 2,
                  'glomerulus_to_golgi': 1, 'glomerulus_to_granule': 1, 'golgi_to_granule': 2, 'golgi_to_golgi': 2,
                  'parallel_fiber_to_basket': 1, 'parallel_fiber_to_golgi': 3, 'parallel_fiber_to_purkinje': 1,
                  'parallel_fiber_to_stellate': 1, 'stellate_to_purkinje': 2, 'stellate_to_stellate': 2,
                  'basket_to_basket': 2, 'purkinje_to_dcn_glut_large': 2, 'mossy_to_dcn_glut_large': 1,
                  'purkinje_to_dcn_GABA': 2, \
                  'io_to_purkinje': 3, 'io_to_basket': 3, 'io_to_stellate': 3, 'io_to_dcn_glut_large': 1,
                  'io_to_dcn_GABA': 1, 'dcn_GABA_to_io': 2}

# Connection pre and post-synaptic neurons
conn_pre_post = {'mossy_to_glomerulus': {'pre': 'mossy_fibers', 'post': 'glomerulus'}, \
                 'ascending_axon_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                 'ascending_axon_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                 'basket_to_purkinje': {'pre': 'basket_cell', 'post': 'purkinje_cell'}, \
                 'glomerulus_to_golgi': {'pre': 'glomerulus', 'post': 'golgi_cell'}, \
                 'glomerulus_to_granule': {'pre': 'glomerulus', 'post': 'granule_cell'}, \
                 'golgi_to_granule': {'pre': 'golgi_cell', 'post': 'granule_cell'}, \
                 'golgi_to_golgi': {'pre': 'golgi_cell', 'post': 'golgi_cell'}, \
                 'parallel_fiber_to_basket': {'pre': 'granule_cell', 'post': 'basket_cell'}, \
                 'parallel_fiber_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                 'parallel_fiber_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                 'parallel_fiber_to_stellate': {'pre': 'granule_cell', 'post': 'stellate_cell'}, \
                 'stellate_to_purkinje': {'pre': 'stellate_cell', 'post': 'purkinje_cell'}, \
                 'basket_to_basket': {'pre': 'basket_cell', 'post': 'basket_cell'}, \
                 'stellate_to_stellate': {'pre': 'stellate_cell', 'post': 'stellate_cell'}, \
                 'mossy_to_dcn_glut_large': {'pre': 'mossy_fibers', 'post': 'dcn_cell_glut_large'}, \
                 'purkinje_to_dcn_glut_large': {'pre': 'purkinje_cell', 'post': 'dcn_cell_glut_large'}, \
                 'purkinje_to_dcn_GABA': {'pre': 'purkinje_cell', 'post': 'dcn_cell_GABA'}, \
                 'io_to_purkinje': {'pre': 'io_cell', 'post': 'purkinje_cell'}, \
                 'io_to_basket': {'pre': 'io_cell', 'post': 'basket_cell'}, \
                 'io_to_stellate': {'pre': 'io_cell', 'post': 'stellate_cell'}, \
                 'io_to_dcn_glut_large': {'pre': 'io_cell', 'post': 'dcn_cell_glut_large'},
                 'io_to_dcn_GABA': {'pre': 'io_cell', 'post': 'dcn_cell_GABA'},
                 'dcn_GABA_to_io': {'pre': 'dcn_cell_GABA', 'post': 'io_cell'}}

neuron_types_to_region = {'golgi_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'granule_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'purkinje_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'basket_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'stellate_cell': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'dcn_cell_glut_large': ['Right Cerebellar Nuclei', 'Left Cerebellar Nuclei'],
                          'dcn_cell_GABA': ['Right Cerebellar Nuclei', 'Left Cerebellar Nuclei'],
                          'io_cell': ['Right Inferior olivary complex', 'Left Inferior olivary complex'],
                          'glomerulus': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'mossy_fibers': ['Right Ansiform lobule', 'Left Ansiform lobule'],
                          'parrot_medulla': ['Right Principal sensory nucleus of the trigeminal',
                                             'Left Principal sensory nucleus of the trigeminal'],
                          'parrot_ponssens': ['Right Pons Sensory', 'Left Pons Sensory']
                          }


def split_mossy_fibers(start_id_scaffold, f=None):
    if f is None:
        f = h5py.File(config.CEREB_SCAFFOLD_PATH, 'r+')
    # We all this to find the indices of the target mossy fibers!:
    # Localized CS
    r_x, r_z = 75, 50
    gloms_pos = np.array(f['cells/placement/glomerulus/positions'])
    x_s_c, x_p_c, z_c = 75., 225., 100.

    # Find glomeruli falling into the selected volume
    target_gloms_bool = np.add(((gloms_pos[:, [0]] - x_s_c) ** 2) / r_x ** 2,
                               ((gloms_pos[:, [2]] - z_c) ** 2) / r_z ** 2).__lt__(1)  # ellipse equation
    target_gloms_id_scaffold_spinal = np.array(np.where(target_gloms_bool)[0] + start_id_scaffold['glomerulus'])

    target_gloms_bool = np.add(((gloms_pos[:, [0]] - x_p_c) ** 2) / r_x ** 2,
                               ((gloms_pos[:, [2]] - z_c) ** 2) / r_z ** 2).__lt__(1)  # ellipse equation
    target_gloms_id_scaffold_principal = np.array(np.where(target_gloms_bool)[0] + start_id_scaffold['glomerulus'])

    conn_glom_mf = np.array(f['cells/connections/mossy_to_glomerulus'])
    # Select the corresponding original MFs
    target_mfs_id_scaffold_spinal = conn_glom_mf[np.isin(conn_glom_mf[:, 1], target_gloms_id_scaffold_spinal), 0]
    target_mfs_id_scaffold_principal = conn_glom_mf[
        np.isin(conn_glom_mf[:, 1], target_gloms_id_scaffold_principal), 0]
    return target_mfs_id_scaffold_spinal, target_mfs_id_scaffold_principal


def get_medulla_mossy_targets(region_mf, neuron_models, start_id_scaffold, target_mfs_id_scaffold_spinal):
    # translate to NEST ids
    target_mfs_id_nest_spinal = target_mfs_id_scaffold_spinal - start_id_scaffold['mossy_fibers'] + \
                                neuron_models['mossy_fibers'][region_mf][0]
    target_mfs_id_nest_spinal = target_mfs_id_nest_spinal.astype(int)

    # Obtain an ordered list of non-duplicates
    return sorted(list(set(target_mfs_id_nest_spinal)))  # Medulla


def get_ponnsess_mossy_targets(region_mf, neuron_models, start_id_scaffold, target_mfs_id_scaffold_principal):
    # translate to NEST ids
    target_mfs_id_nest_principal = target_mfs_id_scaffold_principal - start_id_scaffold['mossy_fibers'] + \
                                   neuron_models['mossy_fibers'][region_mf][0]
    target_mfs_id_nest_principal = target_mfs_id_nest_principal.astype(int)
    return sorted(list(set(target_mfs_id_nest_principal)))  # PONS Sensory


def build_NEST_network(config=None):

    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
    from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
    from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
    from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
    from tvb_multiscale.core.spiking_models.devices import DeviceSet
    from tvb_multiscale.tvb_nest.nest_models.devices import NESTSpikeRecorder  # , NESTMultimeter
    from tvb_multiscale.tvb_nest.nest_models.devices import NESTPoissonGenerator
    from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel

    config = assert_config(config, return_plotter=False)

    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = load_pickled_dict(sim_serial_filepath)

    # Load NEST and use defaults to configure its kernel:
    nest = configure_nest_kernel(load_nest(config=config), config)

    if 'eglif_cond_alpha_multisyn' not in nest.Models():
        try:
            if config.VERBOSE:
                print("Installing cereb module...")
            nest.Install('cerebmodule')
        except:
            if config.VERBOSE:
                print("FAILED! Needing to compile it first!")
            import subprocess
            pwd = os.getcwd()
            cereb_path = '/home/docker/packages/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules/cereb'
            os.chdir(os.path.join(cereb_path, 'build'))
            # This is our shell command, executed by Popen.
            if config.VERBOSE:
                print("Compiling cereb module...")
            p = subprocess.Popen("cmake -Dwith-nest=/home/docker/env/neurosci/nest_build/bin/nest-config ..; make; make install",
                                 stdout=subprocess.PIPE, shell=True)
            if config.VERBOSE:
                print(p.communicate())
                print("Installing cereb module...")
            nest.Install('cerebmodule')
            os.chdir(pwd)

    ###################### NEST simulation parameters #########################################
    TOT_DURATION = config.SIMULATION_LENGTH  # ms
    BACKGROUND_FREQ = 4.
    STIM_FREQ = config.STIMULUS_RATE
    STIM_AMPLITUDE = config.NEST_STIMULUS
    STIM_RATE = 0.

    high_iomli = 120.0  # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
    min_iomli = 40.0

    ######################## NEST simulation setup ##########################################
    # First configure NEST kernel:
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({"overwrite_files": True, "data_path": "sim_data/", "resolution": 0.05})

    if config.VERBOSE:
        print("Building NESTNetwork...")

    # Create NEST network...
    nest_network = NESTNetwork(nest)

    # Load file with positions and connections data
    f = h5py.File(config.CEREB_SCAFFOLD_PATH, 'r+')

    neuron_types = list(f['cells/placement'].keys())
    if config.VERBOSE > 1:
        print(neuron_types)

    neuron_number = {}
    start_id_scaffold = {}

    # Create a dictionary; keys = cell names, values = lists to store neuron models
    neuron_models = {key: [] for key in neuron_types}

    # ...starting from neuronal populations located at specific brain regions...
    nest_network.brain_regions = NESTBrain()

    nest_nodes_inds = []

    if config.NEST_PERIPHERY is False:
        del neuron_types_to_region['parrot_medulla']
        del neuron_types_to_region['parrot_ponssens']

    # All cells are modelled as E-GLIF models;
    # with the only exception of Glomeruli and Mossy Fibers (not cells, just modeled as
    # relays; i.e., parrot neurons)
    neuron_types.remove('dcn_cell_Gly-I')
    for neuron_name in neuron_types:
        pop = neuron_name
        if neuron_name != 'glomerulus' and neuron_name != 'mossy_fibers':
            if neuron_name not in nest.Models():
                nest.CopyModel('eglif_cond_alpha_multisyn', neuron_name)
                nest.SetDefaults(neuron_name, neuron_param[neuron_name])
        else:
            if neuron_name not in nest.Models():
                nest.CopyModel('parrot_neuron', neuron_name)

        neuron_number[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[1]
        start_id_scaffold[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[0]

        neuron_models[neuron_name] = {}
        region_names = neuron_types_to_region[neuron_name]
        nodes_inds = []
        for region in region_names:
            neuron_models[neuron_name][region] = nest.Create(neuron_name, neuron_number[neuron_name])
            if region not in nest_network.brain_regions:
                nest_network.brain_regions[region] = NESTRegionNode(label=region)
                nodes_inds.append(np.where(sim_serial['connectivity.region_labels'] == region)[0][0])
            nest_network.brain_regions[region][pop] = \
                NESTPopulation(neuron_models[neuron_name][region],  # possible NEST model params as well here
                               nest, label=pop, brain_region=region)
            if config.VERBOSE > 1:
                print("\n...created: %s..." % nest_network.brain_regions[region][pop].summary_info())
        nest_nodes_inds += nodes_inds

    if config.NEST_PERIPHERY:
        # We all this to find the indices of the target mossy fibers!:
        target_mfs_id_scaffold_spinal, target_mfs_id_scaffold_principal = split_mossy_fibers(start_id_scaffold, f)
        n_mossy_fibers_medulla = len(target_mfs_id_scaffold_spinal)
        n_mossy_fibers_ponssens = len(target_mfs_id_scaffold_principal)
        nodes_inds = []
        for pop, n_neurons in zip(["parrot_medulla", "parrot_ponssens"],
                                  [n_mossy_fibers_medulla, n_mossy_fibers_ponssens]):
            region_names = neuron_types_to_region[pop]
            for region in region_names:
                if region not in nest_network.brain_regions:
                    nest_network.brain_regions[region] = NESTRegionNode(label=region)
                    nodes_inds.append(np.where(sim_serial['connectivity.region_labels'] == region)[0][0])
                nest_network.brain_regions[region][pop] = \
                    NESTPopulation(nest.Create("parrot_neuron", n_neurons),  # possible NEST model params as well here
                                   nest, label=pop, brain_region=region)
                if config.VERBOSE > 1:
                    print("\n...created: %s..." % nest_network.brain_regions[region][pop].summary_info())

        nest_nodes_inds += nodes_inds

    ### Load connections from hdf5 file and create them in NEST:

    for conn_name in conn_weights.keys():
        conn = np.array(f['cells/connections/' + conn_name])
        pre_name = conn_pre_post[conn_name]["pre"]
        post_name = conn_pre_post[conn_name]["post"]

        for pre_region, post_region in zip(neuron_models[pre_name].keys(), neuron_models[post_name].keys()):
            source = np.array(conn[:, 0] - start_id_scaffold[pre_name] + neuron_models[pre_name][pre_region][0])
            target = np.array(conn[:, 1] - start_id_scaffold[post_name] + neuron_models[post_name][post_region][0])
            pre = list(source.astype(int))
            post = list(target.astype(int))
            if config.VERBOSE > 1:
                print("Connecting  ", conn_name, "!")
                print("%s - %s -> %s -> %s" % (pre_name, pre_region, post_name, post_region))

            if conn_name == "mossy_to_glomerulus":
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * [conn_weights[conn_name]],
                             "delay": np.ones(len(pre)) * conn_delays[conn_name]}
            elif conn_name == "io_bc" or conn_name == "io_sc":
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * conn_weights[conn_name], \
                             "delay": {'distribution': 'exponential_clipped_to_boundary', 'low': min_iomli,
                                       'high': high_iomli, 'lambda': conn_delays[conn]},
                             "receptor_type": conn_receptors[conn_name]}
            else:
                syn_param = {"synapse_model": "static_synapse",
                             "weight": np.ones(len(pre)) * [conn_weights[conn_name]],
                             "delay": np.ones(len(pre)) * conn_delays[conn_name],
                             "receptor_type": conn_receptors[conn_name]}

            nest.Connect(pre, post, {"rule": "one_to_one"}, syn_param)

    if config.NEST_PERIPHERY:
        pop = "parrot_medulla"
        mossy_fibers_medulla = {}
        for region, region_mf in zip(['Right Principal sensory nucleus of the trigeminal',
                                      'Left Principal sensory nucleus of the trigeminal'],
                                     ['Right Ansiform lobule', 'Left Ansiform lobule']):
            if config.VERBOSE > 1:
                print("Connecting! %s - %s -> %s -> %s" % (pop, region, "mossy_fibers", region_mf))
            # translate to NEST ids
            mossy_fibers_medulla[region] = \
                get_medulla_mossy_targets(region_mf, neuron_models,
                                          start_id_scaffold, target_mfs_id_scaffold_spinal)  # Medulla
            nest.Connect(nest_network.brain_regions[region][pop].nodes, mossy_fibers_medulla[region])

        pop = "parrot_ponssens"
        mossy_fibers_ponssens = {}
        for region, region_mf in zip(['Right Pons Sensory', 'Left Pons Sensory'],
                                     ['Right Ansiform lobule', 'Left Ansiform lobule']):
            if config.VERBOSE > 1:
                print("Connecting!  %s - %s -> %s -> %s" % (pop, region, "mossy_fibers", region_mf))
            # translate to NEST ids
            mossy_fibers_ponssens[region] = \
                get_ponnsess_mossy_targets(region_mf, neuron_models,
                                           start_id_scaffold, target_mfs_id_scaffold_principal)  # PONS Sensory
            nest.Connect(nest_network.brain_regions[region][pop].nodes, mossy_fibers_ponssens[region])

    # Background noise input device as Poisson process
    nest_network.input_devices["Background"] = DeviceSet(label="Background", model="poisson_generator")
    for region in ['Right Ansiform lobule', 'Left Ansiform lobule']:
        nest_network.input_devices["Background"][region] = \
            NESTPoissonGenerator(nest.Create('poisson_generator',
                                             params={'rate': BACKGROUND_FREQ, 'start': 0.0, 'stop': TOT_DURATION}),
                                 nest, model="poisson_generator",
                                 label="Background", brain_region=region)
        nest.Connect(nest_network.input_devices["Background"][region].device,
                     neuron_models['mossy_fibers'][region])
        if config.VERBOSE > 1:
            print("Connected!  %s - %s -> %s -> %s" % ("Background", region, pop, region))

    if config.NEST_PERIPHERY:
        # Whisking stimulus input device as sinusoidally modulated Poisson process
        pop = 'parrot_medulla'
        nest_network.input_devices["Stimulus"] = DeviceSet(label="Stimulus", model="sinusoidal_poisson_generator")
        for region in ['Right Principal sensory nucleus of the trigeminal',
                       'Left Principal sensory nucleus of the trigeminal']:
            nest_network.input_devices["Stimulus"][region] = \
                NESTPoissonGenerator(nest.Create('sinusoidal_poisson_generator',
                                                 params={"rate": STIM_RATE, "amplitude": STIM_AMPLITUDE,
                                                         "frequency": STIM_FREQ, "phase": 0.0}),
                                     nest, model="sinusoidal_poisson_generator",
                                     label="Stimulus", brain_region=region)
            nest.Connect(nest_network.input_devices["Stimulus"][region].device,
                         nest_network.brain_regions[region][pop].nodes)
            if config.VERBOSE > 1:
                print("Connected!  %s - %s -> %s -> %s" % ("Stimulus", region, pop, region))

    # Create output, measuring devices, spike_recorders and multimeters measuring V_m:
    params_spike_recorder = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
    params_spike_recorder["record_to"] = "ascii"
    # params_multimeter = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"].copy()
    # params_multimeter["record_to"] = "ascii"
    # params_multimeter["interval"] = 1.0
    for pop, regions in neuron_types_to_region.items():
        # pop_ts = "%s_ts" % pop
        nest_network.output_devices[pop] = DeviceSet(label=pop, model="spike_recorder")

        for region in regions:
            nest_network.output_devices[pop][region] = \
                NESTSpikeRecorder(nest.Create("spike_recorder", 1, params=params_spike_recorder),
                                  nest, model="spike_recorder", label=pop, brain_region=region)
            if pop == "granule_cell":
                nodes = nest_network.brain_regions[region][pop].nodes[0::10]
            else:
                nodes = nest_network.brain_regions[region][pop].nodes
            nest.Connect(nodes, nest_network.output_devices[pop][region].device)
            nest_network.output_devices[pop].update()  # update DeviceSet after the new NESTDevice entry
            if config.VERBOSE > 1:
                print("\n...created spike_recorder device for population %s in brain region %s..." % (pop, region))

        # if pop not in ['mossy_fibers', "whisking_stimulus"]:
        #     nest_network.output_devices[pop_ts] = DeviceSet(label=pop_ts, model="multimeter")
        #     # Create and connect population multimeter for this region:
        #     nest_network.output_devices[pop_ts][region] = \
        #         NESTMultimeter(nest.Create("multimeter", 1, params=params_multimeter),
        #                        nest, model="multimeter", label=pop_ts, brain_region=region)
        #     nest.Connect(nest_network.output_devices[pop_ts][region].device,
        #                  nest_network.brain_regions[region][pop].nodes)
        #     nest_network.output_devices[pop_ts].update()  # update DeviceSet after the new NESTDevice entry
        #     if config.VERBOSE > 1:
        #         print("\n...created multimeter device for population %s in brain region %s..." % (pop_ts, region))

    nest_network.configure()
    if config.VERBOSE > 1:
        nest_network.print_summary_info_details(recursive=1, connectivity=False)

    return nest_network, nest_nodes_inds, neuron_models, neuron_number


def plot_nest_results_raster(nest_network, neuron_models, neuron_number, config):

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    goc_events = nest_network.output_devices['golgi_cell']['Left Ansiform lobule'].events
    goc_evs = goc_events['senders']
    goc_times = goc_events['times']

    grc_events = nest_network.output_devices['granule_cell']['Left Ansiform lobule'].events
    grc_evs = grc_events['senders']
    grc_times = grc_events['times']

    glom_events = nest_network.output_devices['glomerulus']['Left Ansiform lobule'].events
    glom_evs = glom_events['senders']
    glom_times = glom_events['times']

    pc_events = nest_network.output_devices['purkinje_cell']['Left Ansiform lobule'].events
    pc_evs = pc_events['senders']
    pc_times = pc_events['times']

    sc_events = nest_network.output_devices['stellate_cell']['Left Ansiform lobule'].events
    sc_evs = sc_events['senders']
    sc_times = sc_events['times']

    bc_events = nest_network.output_devices['basket_cell']['Left Ansiform lobule'].events
    bc_evs = bc_events['senders']
    bc_times = bc_events['times']

    io_events = nest_network.output_devices['io_cell']['Left Inferior olivary complex'].events
    io_evs = io_events['senders']
    io_times = io_events['times']

    dcng_events = nest_network.output_devices['dcn_cell_GABA']['Left Cerebellar Nuclei'].events
    dcng_evs = dcng_events['senders']
    dcng_times = dcng_events['times']

    dcn_events = nest_network.output_devices['dcn_cell_glut_large']['Left Cerebellar Nuclei'].events
    dcn_evs = dcn_events['senders']
    dcn_times = dcn_events['times']


    # ######################### PLOTTING PSTH AND RASTER PLOTS ########################

    CELL_TO_PLOT = ['glomerulus', 'granule_cell', 'basket_cell', 'stellate_cell', 'purkinje_cell',
                    'io_cell', 'dcn_cell_GABA',  'dcn_cell_glut_large']

    cells = {'granule_cell': [grc_times, grc_evs],
             'golgi_cell': [goc_times, goc_evs],
             'glomerulus': [glom_times, glom_evs],
             'purkinje_cell': [pc_times, pc_evs],
             'stellate_cell': [sc_times, sc_evs],
             'basket_cell': [bc_times, bc_evs],
             'io_cell': [io_times, io_evs],
             'dcn_cell_GABA': [dcng_times, dcng_evs],
             'dcn_cell_glut_large': [dcn_times, dcn_evs]}

    color = {'granule_cell': '#E62214',  # 'rgba(255, 0, 0, .8)',
             'golgi_cell': '#332EBC',  # 'rgba(0, 255, 0, .8)',
             'glomerulus': '#0E1030',  # rgba(0, 0, 0, .8)',
             'purkinje_cell': '#0F8944',  # 'rgba(64, 224, 208, .8)',
             'stellate_cell': '#FFC425',  # 'rgba(234, 10, 142, .8)',
             'basket_cell': '#F37735',
             'io_cell': 'rgba(75, 75, 75, .8)',
             'dcn_cell_GABA': 'rgba(100, 100, 100, .8)',
             'dcn_cell_glut_large': '#080808'}  # 'rgba(234, 10, 142, .8)'}

    # PSTH

    def metrics(spikeData, TrialDuration, cell, figure_handle, sel_row):
        id_spikes = np.sort(np.unique(spikeData, return_index=True))
        bin_size = 5  # [ms]
        n_bins = int(TrialDuration / bin_size) + 1
        psth, tms = np.histogram(spikeData, bins=n_bins, range=(0, TrialDuration))

        # absolute frequency
        abs_freq = np.zeros(id_spikes[0].shape[0])
        for idx, i in enumerate(id_spikes[0]):
            count = np.where(spikeData == i)[0]
            abs_freq[idx] = count.shape[0]

        # mean frequency
        m_f = (id_spikes[0].shape[0]) / ((TrialDuration / 1000) * len(neuron_models[cell]))

        layout = go.Layout(
            scene=dict(aspectmode='data'),
            xaxis={'title': 'time (ms)'},
            yaxis={'title': 'number of spikes'}
        )

        n_neurons = neuron_number[cell]
        if cell == "granule_cell":
            n_neurons = int(np.round(n_neurons/10))
        figure_handle.add_trace(go.Bar(
            x=tms[0:len(tms) - 1],
            y=psth / ((bin_size * 0.001) * n_neurons),
            width=4.0,
            marker=dict(
                color=color[cell])
        ), row=sel_row, col=1)

        if config.VERBOSE > 1:
            print("mean frequency: ", int(m_f))

        return tms

    # RASTER
    def raster(times, cell_ids, cell, fig_handle, sel_row):
        trace0 = go.Scatter(
            x=times,
            y=cell_ids,
            name='',
            mode='markers',
            marker=dict(
                size=4,
                color=color[cell],
                line=dict(
                    width=.2,
                    color='rgb(0, 0, 0)'
                )
            )
        )
        fig_handle.add_trace(trace0, row=sel_row, col=1)

    fig_psth = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                             y_title='Frequency [Hz]')
    fig_raster = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                               y_title='# cells')
    num = 1
    for c in CELL_TO_PLOT:
        times = cells[c][0]
        cell_ids = cells[c][1]
        metrics(times, config.SIMULATION_LENGTH, c, fig_psth, num)
        raster(times, cell_ids, c, fig_raster, num)
        num += 1
    fig_psth.update_xaxes(range=[0, config.SIMULATION_LENGTH * 1.1])
    fig_raster.update_xaxes(range=[0, config.SIMULATION_LENGTH * 1.1])
    fig_psth.update_layout(showlegend=False)
    fig_raster.update_layout(showlegend=False)
    if config.figures.SAVE_FLAG:
        try:
            fig_psth.write_image(os.path.join(config.figures.FOLDER_FIGURES, "NESTpsth.%s" % config.figures.FIG_FORMAT))
            fig_raster.write_image(os.path.join(config.figures.FOLDER_FIGURES, "NESTraster.%s" % config.figures.FIG_FORMAT))
        except Exception as e:
            warnings.warn("Failed to write_image for plotly figures with error:\n" % str(e))
    if config.figures.SHOW_FLAG:
        fig_psth.show()
        fig_raster.show()
    else:
        # TODO: find a better way to delete plotly figures
        # The current one is taken from here: https://community.plotly.com/t/remove-all-traces/13469
        # There might not be a better one yet...: https://github.com/plotly/plotly.py/issues/2725
        fig_psth.data = []
        fig_raster.data = []
        fig_psth.layout = {}
        fig_raster.layout = {}
        fig_psth = None
        fig_raster = None
    return fig_psth, fig_raster


def simulate_nest_network(nest_network, config, neuron_models={}, neuron_number={}):
    tic = time.time()
    # Simulate:
    if config.VERBOSE:
        print("\nSimulating NEST network...")
    nest_network.nest_instance.Simulate(config.SIMULATION_LENGTH)
    if config.VERBOSE:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    return nest_network


def run_nest_workflow(PSD_target=None, model_params={}, config=None, **config_args):
    tic = time.time()
    plot_flag = config_args.get('plot_flag', DEFAULT_ARGS.get('plot_flag'))
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    if config.VERBOSE:
        print("\n\n------------------------------------------------\n\n"+
              "Running NEST workflow for plot_flag=%s, \nand model_params=\n%s...\n" 
              % (str(plot_flag), str(config.model_params)))
    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Build the NEST network
    nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
    # Simulate the NEST network
    nest_network = simulate_nest_network(nest_network, config, neuron_models, neuron_number)
    # Plot results
    if plotter is not None:
        from examples.plot_write_results import plot_write_spiking_network_results
        simulation_length, transient = configure_simulation_length_with_transient(config)
        plot_write_spiking_network_results(nest_network, connectivity=connectivity,
                                           time=None, transient=transient, monitor_period=simulator.monitors[0].period,
                                           plot_per_neuron=False, plotter=plotter, writer=None, config=config)
        plot_nest_results_raster(nest_network, neuron_models, neuron_number, config)
    if config.VERBOSE:
        print("\nFinished NEST workflow in %g sec!\n" % (time.time() - tic))
    results = {"nest_network": nest_network, "simulator": simulator, "config": config}
    return results


if __name__ == "__main__":
    parser = args_parser("nest_script")
    args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
    verbose = args.get('verbose', DEFAULT_ARGS['verbose'])
    if verbose:
        print("Running %s with arguments:\n" % parser.description)
        print(args, "\n")
    run_nest_workflow(**args)

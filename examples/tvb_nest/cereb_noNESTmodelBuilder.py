# -*- coding: utf-8 -*-
import time
import os
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import CONFIGURED, Config
from examples.tvb_nest.example import results_path_fun
from examples.plot_write_results import plot_write_results
from tvb_multiscale.tvb_nest.nest_models.builders.models.cereb import CerebBuilder
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder, nest_nodes_ids,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 transient=0.0, use_numba=True, variables_of_interest=None,
                 config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(
                    output_base=results_path_fun(nest_model_builder, tvb_nest_builder, tvb_to_nest_mode, nest_to_tvb,
                                                 config))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = CoSimulatorBuilder()
    simulator_builder.use_numba = use_numba
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.variables_of_interest = variables_of_interest
    simulator_builder.connectivity = connectivity
    simulator_builder.delays_flag = delays_flag
    simulator = simulator_builder.build(**model_params)

    # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    print("Building NEST network...")
    tic = time.time()

    def remove_files():
        for f in os.listdir('.'):
            if '.gdf' in f or '.dat' in f:
                os.remove(f)

    remove_files()

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = \
        nest_model_builder(simulator, nest_nodes_ids,
                           os.path.join(os.getcwd().split("tvb_nest")[0],
                                        "tvb_nest", "../data", "cerebellar_cortex_scaffold.hdf5"),
                           config=config, set_defaults=True)
    nest_model_builder.modules_to_install = ["cereb"]
    nest_model_builder.compile_install_nest_modules(["cereb"])

    # # Common order of neurons' number per population:
    # nest_network = nest_model_builder.build_spiking_network()

    # or...

    # ----------------------------------------------------------------------------------------------------------------
    # ----Uncomment below to build the NEST network without using the NEST model builder of TVB-NEST------------------
    # ----------------------------------------------------------------------------------------------------------------
    import h5py

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
                       'E_L': -62.0, 'Vmin': -150.0,
                       'lambda_0': 1.0, 'tau_V': 0.4, 'I_e': 16.214, 'kadap': 0.217, 'k1': 0.031, 'k2': 0.023,
                       'A1': 259.988, 'A2': 178.01,
                       'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['golgi'],
                       'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
        'granule_cell': {'t_ref': 1.5, 'C_m': 7.0, 'tau_m': 24.15, 'V_th': -41.0, 'V_reset': -70.0, 'Vinit': -62.0,
                         'E_L': -62.0, 'Vmin': -150.0,
                         'lambda_0': 1.0, 'tau_V': 0.3, 'I_e': -0.888, 'kadap': 0.022, 'k1': 0.311, 'k2': 0.041,
                         'A1': 0.01, 'A2': -0.94,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['granule'],
                         'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
        'purkinje_cell': {'t_ref': 0.5, 'C_m': 334.0, 'tau_m': 47.0, 'V_th': -43.0, 'V_reset': -69.0, 'Vinit': -59.0,
                          'E_L': -59.0,
                          'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 742.54, 'kadap': 1.492, 'k1': 0.1950, 'k2': 0.041,
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
                          'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli}}

    # Connection weights
    conn_weights = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 0.05, 'ascending_axon_to_purkinje': 0.175,
                    'basket_to_purkinje': 3.638, \
                    'glomerulus_to_golgi': 0.0125, 'glomerulus_to_granule': 0.361, 'golgi_to_granule': 0.338, \
                    'parallel_fiber_to_basket': 0.002, 'parallel_fiber_to_golgi': 0.008,
                    'parallel_fiber_to_purkinje': 0.044, \
                    'parallel_fiber_to_stellate': 0.003, 'stellate_to_purkinje': 1.213}

    # Connection delays
    conn_delays = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 2.0, 'ascending_axon_to_purkinje': 2.0,
                   'basket_to_purkinje': 4.0,
                   'glomerulus_to_golgi': 4.0, 'glomerulus_to_granule': 4.0, 'golgi_to_granule': 2.0,
                   'parallel_fiber_to_basket': 5.0, 'parallel_fiber_to_golgi': 5.0, 'parallel_fiber_to_purkinje': 5.0,
                   'parallel_fiber_to_stellate': 5.0, 'stellate_to_purkinje': 5.0}

    # Connection receptors
    conn_receptors = {'ascending_axon_to_golgi': 3, 'ascending_axon_to_purkinje': 1, 'basket_to_purkinje': 2,
                      'glomerulus_to_golgi': 1, 'glomerulus_to_granule': 1, 'golgi_to_granule': 2,
                      'parallel_fiber_to_basket': 1, 'parallel_fiber_to_golgi': 3, 'parallel_fiber_to_purkinje': 1,
                      'parallel_fiber_to_stellate': 1, 'stellate_to_purkinje': 2}

    # Connection pre and post-synaptic neurons
    conn_pre_post = {'mossy_to_glomerulus': {'pre': 'mossy_fibers', 'post': 'glomerulus'}, \
                     'ascending_axon_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                     'ascending_axon_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                     'basket_to_purkinje': {'pre': 'basket_cell', 'post': 'purkinje_cell'}, \
                     'glomerulus_to_golgi': {'pre': 'glomerulus', 'post': 'golgi_cell'}, \
                     'glomerulus_to_granule': {'pre': 'glomerulus', 'post': 'granule_cell'}, \
                     'golgi_to_granule': {'pre': 'golgi_cell', 'post': 'granule_cell'}, \
                     'parallel_fiber_to_basket': {'pre': 'granule_cell', 'post': 'basket_cell'}, \
                     'parallel_fiber_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'}, \
                     'parallel_fiber_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'}, \
                     'parallel_fiber_to_stellate': {'pre': 'granule_cell', 'post': 'stellate_cell'}, \
                     'stellate_to_purkinje': {'pre': 'stellate_cell', 'post': 'purkinje_cell'}}

    # Get the nest instance of the NEST model builder:
    nest = nest_model_builder.nest_instance
    nest_model_builder._configure_nest_kernel()

    # Prepare the NEST kernel:
    # nest.ResetKernel()
    # # nest.set_verbosity('M_ERROR')
    # # nest.SetKernelStatus({"local_num_threads" : cpu_count()})

    # Load the network source file:
    f = h5py.File(nest_model_builder.path_to_network_source_file, 'r+')

    neuron_types = list(f['cells/placement'].keys())
    print(neuron_types)

    neuron_number = {}
    start_id_scaffold = {}

    # ### Creating population of neurons
    # Instantiate conductance based Extended-Generalized Leaky Integrate and Fire models (E-GLIF) for each cell type.
    # The only exception is represented by Glomeruli and Mossy Fibers; these are not actual cells but just 'relays',
    # used to deliver input spikes to other cells. Here, Glomeruli are created as *parrot_neurons*

    # Create a dictionary; keys = cell names, values = lists to store neuron models
    neuron_models = {key: [] for key in neuron_types}

    # All cells are modelled as E-GLIF models;
    # with the only exception of Glomeruli and Mossy Fibers (not cells, just modeled as
    # relays; i.e., parrot neurons)
    from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
    from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
    from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
    nest_region_label = simulator.connectivity.region_labels[nest_nodes_ids[0]]
    nest_brain = NESTBrain(nest_instance=nest)
    nest_brain[nest_region_label] = NESTRegionNode(label=nest_region_label, nest_instance=nest)
    for neuron_name in neuron_types:
        if neuron_name != 'glomerulus' and neuron_name != 'mossy_fibers':
            if neuron_name not in nest.Models():
                model = 'eglif_cond_alpha_multisyn'
                nest.CopyModel('eglif_cond_alpha_multisyn', neuron_name)
                nest.SetDefaults(neuron_name, neuron_param[neuron_name])
        else:
            model = 'parrot_neuron'
            if neuron_name not in nest.Models():
                nest.CopyModel('parrot_neuron', neuron_name)

        neuron_number[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[1]
        start_id_scaffold[neuron_name] = np.array(f['cells/placement/' + neuron_name + '/identifiers'])[0]

        neuron_models[neuron_name] = nest.Create(neuron_name, neuron_number[neuron_name])

        # Set the neurons' indices into a NESTPopulation class instance inside the NESTBrain class instance:
        nest_brain[nest_region_label][neuron_name] = NESTPopulation(neuron_models[neuron_name],
                                                                    label=neuron_name, model=model, nest_instance=nest)

    # ### Creating synaptic connections
    # Here we create synaptic connections among neurons.
    # A message will be printed below the next cell when each connection type is done:

    # ### Load connections from hdf5 file and create them in NEST:
    for conn in conn_weights.keys():
        conn_name = conn
        conn = np.array(f['cells/connections/'+conn_name])
        pre_name = conn_pre_post[conn_name]["pre"]
        post_name = conn_pre_post[conn_name]["post"]
        pre = [int(x-start_id_scaffold[pre_name]+neuron_models[pre_name][0]) for x in conn[:,0]]
        post = [int(x-start_id_scaffold[post_name]+neuron_models[post_name][0]) for x in conn[:,1]]
        if conn_name == "mossy_to_glomerulus":
            syn_param = {"model": "static_synapse", "weight": conn_weights[conn_name], "delay": conn_delays[conn_name]}
        else:
            syn_param = {"model": "static_synapse", "weight": conn_weights[conn_name], "delay": conn_delays[conn_name],
                         "receptor_type": conn_receptors[conn_name]}
        nest.Connect(pre, post, {"rule": "one_to_one"}, syn_param)
        print("Connections ", conn_name, " done!")

    from pandas import Series
    from tvb_multiscale.core.spiking_models.devices import DeviceSet
    from tvb_multiscale.tvb_nest.nest_models.devices import NESTPoissonGenerator, NESTSpikeDetector, NESTMultimeter

    # ### Defining stimuli
    # Into the next cell, the user can define the parameters value for the simulation. The background input is a 4 Hz Poisson process to glomeruli, for 300 ms. Then a 100-Hz burst is provided, lasting 100 ms. The user can set the following parameters:
    # 1. RECORD_VM: by default, spike data are recorded. If you want to record voltage-traces too, please set this variable to 'True', but consider that this is going to increase the computational time of the simulation.
    # 2. TOT_DURATION: duration of whole simulation, in milliseconds.
    # 3. STIM_START: when the burst (a Poisson process spike train) should start.
    # 4. STIM_END : when the burst should stop.
    # 5. STIM_FREQ: frequency of the burst
    RECORD_VM = True
    TOT_DURATION = 600.  # mseconds
    STIM_START = 250.  # beginning of stimulation
    STIM_END = 500.  # end of stimulation
    STIM_FREQ = 100.  # Frequency in Hz
    BACKGROUND_FREQ = 4.

    # Create stimulation devices in NEST and connect to input neural populations (mossy_fibers).
    mossy_fibers_num = len(neuron_models['mossy_fibers'])

    input_devices = Series()

    # STIM as Poisson process
    STIM = nest.Create('poisson_generator', params={'rate': STIM_FREQ, 'start': STIM_START, 'stop': STIM_END})
    # TODO: Find out what this is!!!
    # mossy_fibers_pos = np.array(f['cells/placement/mossy_fibers/positions'])
    # x_c, z_c = 200., 200.
    # Connection to glomeruli
    nest.Connect(STIM, neuron_models['mossy_fibers'])
    # Set it in a TVB-NEST DeviceSet class instance for the specific brain region:
    input_devices["Stimulus"] = DeviceSet(label="Stimulus", model="poisson_generator")
    input_devices["Stimulus"][nest_region_label] = NESTPoissonGenerator(STIM, nest)

    # Background as Poisson process
    background = nest.Create('poisson_generator', params={'rate': BACKGROUND_FREQ, 'start': 0.0})
    nest.Connect(background, neuron_models['mossy_fibers'])
    # Set it in a TVB-NEST DeviceSet class instance for the specific brain region:
    input_devices["Background"] = DeviceSet(label="Background", model="poisson_generator")
    input_devices["Background"][nest_region_label] = NESTPoissonGenerator(background, nest)

    # ### Defining recording devices
    output_devices = Series()

    # Create spike detectors and connect them to the cells; if the user selected RECORD_VM, also voltage will be recorded
    ## Record spikes from granule and Golgi cells
    moss_fib_spikes = nest.Create("spike_detector",
                                  params={"withgid": True, "withtime": True, "to_file": False,
                                          "label": "mossy_fibers_spikes"})
    glom_spikes = nest.Create("spike_detector",
                              params={"withgid": True, "withtime": True, "to_file": False, "label": "glomerulus_spikes"})
    grc_spikes = nest.Create("spike_detector",
                             params={"withgid": True, "withtime": True, "to_file": False, "label": "granule_spikes"})
    goc_spikes = nest.Create("spike_detector",
                             params={"withgid": True, "withtime": True, "to_file": False, "label": "golgi_spikes"})
    bc_spikes = nest.Create("spike_detector",
                            params={"withgid": True, "withtime": True, "to_file": False, "label": "basket_spikes"})
    sc_spikes = nest.Create("spike_detector",
                            params={"withgid": True, "withtime": True, "to_file": False, "label": "stellate_spikes"})
    pc_spikes = nest.Create("spike_detector",
                            params={"withgid": True, "withtime": True, "to_file": False, "label": "purkinje_spikes"})

    # Here you can choose which devices you want to connect and thus the neural populations you want to record.
    # Increasing the number of recorded cells can increase the duration of the simulation
    populations_sizes = []
    populations = []
    nest.Connect(neuron_models['mossy_fibers'], moss_fib_spikes)
    populations_sizes.append(len(neuron_models['mossy_fibers']))
    populations.append('mossy_fibers')
    nest.Connect(neuron_models['glomerulus'], glom_spikes)
    populations_sizes.append(len(neuron_models['glomerulus']))
    populations.append('glomerulus')
    nest.Connect(neuron_models['granule_cell'], grc_spikes)
    populations_sizes.append(len(neuron_models['granule_cell']))
    populations.append('granule_cell')
    nest.Connect(neuron_models['golgi_cell'], goc_spikes)
    populations_sizes.append(len(neuron_models['golgi_cell']))
    populations.append('golgi_cell')
    nest.Connect(neuron_models['basket_cell'], bc_spikes)
    populations_sizes.append(len(neuron_models['basket_cell']))
    populations.append('basket_cell')
    nest.Connect(neuron_models['stellate_cell'], sc_spikes)
    populations_sizes.append(len(neuron_models['stellate_cell']))
    populations.append('stellate_cell')
    nest.Connect(neuron_models['purkinje_cell'], pc_spikes)
    populations_sizes.append(len(neuron_models['purkinje_cell']))
    populations.append('purkinje_cell')

    # Set them in a TVB-NEST DeviceSet class instance for the specific brain region:
    for device_id, label in zip([moss_fib_spikes, glom_spikes, grc_spikes, goc_spikes, bc_spikes, sc_spikes, pc_spikes],
                                ['mossy_fibers', "glomerulus_spikes", "granule_spikes", "golgi_spikes", "basket_spikes",
                                 "stellate_spikes", "purkinje_spikes"]):
        output_devices[label] = DeviceSet(label=label, model="spike_detector")
        output_devices[label][nest_region_label] = NESTSpikeDetector(device_id, nest)

    if RECORD_VM:
        def neurons_inds_fun(neurons_inds, n_neurons=100):
            # We use this in order to measure up to n_neurons neurons from every population
            n_neurons_inds = len(neurons_inds)
            if n_neurons_inds > n_neurons:
                return tuple(np.array(neurons_inds)[0:-1:int(np.ceil(1.0*n_neurons_inds/n_neurons))])
            else:
                return neurons_inds

        print("Recording membrane voltage")
        grc_vm = nest.Create("multimeter")
        goc_vm = nest.Create("multimeter")
        bc_vm = nest.Create("multimeter")
        sc_vm = nest.Create("multimeter")
        pc_vm = nest.Create("multimeter")

        nest.SetStatus(grc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": False, "label": "granule_vm"})
        nest.SetStatus(goc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": False, "label": "golgi_vm"})
        nest.SetStatus(bc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": False, "label": "basket_vm"})
        nest.SetStatus(sc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": False, "label": "stellate_vm"})
        nest.SetStatus(pc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": False, "label": "purkinje_vm"})

        nest.Connect(grc_vm, neurons_inds_fun(neuron_models['granule_cell']))
        nest.Connect(goc_vm, neurons_inds_fun(neuron_models['golgi_cell']))
        nest.Connect(bc_vm, neurons_inds_fun(neuron_models['basket_cell']))
        nest.Connect(sc_vm, neurons_inds_fun(neuron_models['stellate_cell']))
        nest.Connect(pc_vm, neurons_inds_fun(neuron_models['purkinje_cell']))

    # Set them in a TVB-NEST DeviceSet class instance for the specific brain region:
    for device_id, label in zip([grc_vm, goc_vm, bc_vm, sc_vm, pc_vm],
                                ["granule_vm", "golgi_vm", "basket_vm", "stellate_vm", "purkinje_vm"]):
        output_devices[label] = DeviceSet(label=label, model="multimeter")
        output_devices[label][nest_region_label] = NESTMultimeter(device_id, nest)

    f.close()

    # Finally construct the NEST network model:
    from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
    nest_model_builder._update_default_min_delay()
    nest_network = NESTNetwork(nest_instance=nest,
                               brain_regions=nest_brain,
                               output_devices=output_devices,
                               input_devices=input_devices, config=nest_model_builder.config)

    print(nest_network.print_str(connectivity=True))
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model ---------------------------------------
    if tvb_nest_builder is not None:
        print("Building TVB-NEST interface...")
        tic = time.time()
        # Build a TVB-NEST interface with all the appropriate connections between the
        # TVB and NEST modelled regions
        # Using all default parameters for this example
        tvb_nest_builder = tvb_nest_builder(simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                            populations_sizes=populations_sizes[0])
        tvb_nest_model = tvb_nest_builder.build_interface(tvb_to_nest_mode=tvb_to_nest_mode, nest_to_tvb=nest_to_tvb)
        print(tvb_nest_model.print_str(detailed_output=True, connectivity=False))
        print("Done! in %f min" % ((time.time() - tic)/60))

    # -----------------------------------4. Simulate and gather results-------------------------------------------------
    tvb_results = []
    t_start = time.time()
    if tvb_nest_builder is not None:
        print("Simulating TVB-NEST!...")
        # Configure the simulator with the TVB-NEST interface...
        simulator.configure(tvb_nest_model)
        # ...and simulate!
        tvb_results = simulator.run(simulation_length=TOT_DURATION)
        # Integrate NEST one more NEST time step so that multimeters get the last time point
        # unless you plan to continue simulation later
        simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        simulator.tvb_spikeNet_interface.nest_instance.Cleanup()
    else:
        print("Simulating only NEST!...")
        nest_network.nest_instance.Prepare()
        nest_network.nest_instance.Run(TOT_DURATION + nest_network.nest_instance.GetKernelStatus("resolution"))
        nest_network.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(tvb_results, simulator,
                               spiking_network=nest_network, spiking_nodes_ids=nest_nodes_ids,
                               populations=populations, populations_sizes=populations_sizes,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plot_per_neuron=False, plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    return tvb_results, simulator


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id, label in enumerate(connectivity.region_labels):
        if label.find("cereb") > 0:
            nest_nodes_ids.append(id)

    if len(nest_nodes_ids) == 0:
        nest_nodes_ids = [0]  # if the connectivity doesn't have cerebellum, just set a region for testing
    tvb_model = ReducedWongWangExcIO  # ReducedWongWangExcIOInhI

    model_params = {}

    main_example(tvb_model, CerebBuilder, None, #RedWWexcIOBuilder,  # ,
                 nest_nodes_ids,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True, transient=0.0,
                 variables_of_interest=None, config=None, **model_params)

# -*- coding: utf-8 -*-

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import neuron_types_to_region
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_script import *


def print_available_interfaces():
    # options for a nonopinionated builder:
    from tvb_multiscale.core.interfaces.transformers.models.models import Transformers
    from tvb_multiscale.core.interfaces.transformers.builders import \
        DefaultTVBtoSpikeNetTransformers, DefaultSpikeNetToTVBTransformers, \
        DefaultTVBtoSpikeNetModels, DefaultSpikeNetToTVBModels
    from tvb_multiscale.tvb_nest.interfaces.builders import \
        TVBtoNESTModels, NESTInputProxyModels, DefaultTVBtoNESTModels, \
        NESTtoTVBModels, NESTOutputProxyModels, DefaultNESTtoTVBModels

    def print_enum(enum):
        print("\n", enum)
        for name, member in enum.__members__.items():
            print(name, "= ", member.value)

    print("Available input (NEST->TVB update) / output (TVB->NEST coupling) interface models:")
    print_enum(TVBtoNESTModels)
    print_enum(NESTtoTVBModels)

    print("\n\nAvailable input (spikeNet->TVB update) / output (TVB->spikeNet coupling) transformer models:")

    print_enum(DefaultTVBtoSpikeNetModels)
    print_enum(DefaultTVBtoSpikeNetTransformers)

    print_enum(DefaultSpikeNetToTVBModels)
    print_enum(DefaultSpikeNetToTVBTransformers)

    print("\n\nAvailable input (NEST->TVB update) / output (TVB->NEST coupling) proxy models:")

    print_enum(DefaultTVBtoNESTModels)
    print_enum(NESTInputProxyModels)

    print_enum(NESTOutputProxyModels)
    print_enum(DefaultNESTtoTVBModels)

    print("\n\nAll basic transformer models:")
    print_enum(Transformers)

    from tvb_multiscale.core.interfaces.transformers.models.thalamocortical_wc import \
        DefaultSpikeNetToTVBTransformersThalamoCorticalWCInverseSigmoidal
    print_enum(DefaultSpikeNetToTVBTransformersThalamoCorticalWCInverseSigmoidal)


def build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config):

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions

    #     # ---------------------------- Opinionated TVB<->NEST interface builder----------------------------
    #     from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import WilsonCowanTVBNESTInterfaceBuilder
    #     tvb_spikeNet_model_builder =  WilsonCowanTVBNESTInterfaceBuilder()  # opinionated builder

    # ---------------------------- Non opinionated TVB<->NEST interface builder----------------------------
    from tvb_multiscale.tvb_nest.interfaces.builders import TVBNESTInterfaceBuilder
    # from tvb_multiscale.core.interfaces.transformers.models.thalamocortical_wc import ThalamocorticalWCLinearRate
    from tvb_multiscale.core.interfaces.transformers.models.thalamocortical_wc import \
        DefaultTVBtoSpikeNetTransformersThalamoCorticalWC
    
    tvb_spikeNet_model_builder = TVBNESTInterfaceBuilder()  # non opinionated builder

    tvb_spikeNet_model_builder._tvb_to_spikeNet_transformer_models = DefaultTVBtoSpikeNetTransformersThalamoCorticalWC

    if config.INVERSE_SIGMOIDAL_NEST_TO_TVB:
        # !!! THIS WILL TURN ON THE INVERSE SIGMOIMDAL TRANSFORMER FOR NEST -> TVB INTERFACE !!!
        from tvb_multiscale.core.interfaces.transformers.models.thalamocortical_wc import \
            DefaultSpikeNetToTVBTransformersThalamoCorticalWCInverseSigmoidal

        tvb_spikeNet_model_builder._spikeNet_to_tvb_transformer_models = \
            DefaultSpikeNetToTVBTransformersThalamoCorticalWCInverseSigmoidal

    tvb_spikeNet_model_builder.config = config
    tvb_spikeNet_model_builder.tvb_cosimulator = simulator
    tvb_spikeNet_model_builder.spiking_network = nest_network
    # This can be used to set default tranformer and proxy models:
    tvb_spikeNet_model_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
    tvb_spikeNet_model_builder.input_flag = True  # If True, NEST->TVB update will be implemented
    tvb_spikeNet_model_builder.output_flag = True  # If True, TVB->NEST coupling will be implemented
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / NEST device for each spiking region,
    # "1-to-1" TVB->NEST coupling.
    # If any other value, we need 1 "TVB proxy node" / NEST device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in NEST,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->NEST coupling.
    tvb_spikeNet_model_builder.default_coupling_mode = "TVB"  # "spikeNet" # "TVB"
    tvb_spikeNet_model_builder.proxy_inds = np.array(nest_nodes_inds)
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    tvb_spikeNet_model_builder.exclusive_nodes = True

    tvb_spikeNet_model_builder.output_interfaces = []
    tvb_spikeNet_model_builder.input_interfaces = []

    # from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels

    # if tvb_spikeNet_model_builder.default_coupling_mode == "TVB":
    #     proxy_inds = nest_nodes_inds
    # else:
    #     proxy_inds = np.arange(simulator.connectivity.number_of_regions).astype('i')
    #     proxy_inds = np.delete(proxy_inds, nest_nodes_inds)
    # This is a user defined TVB -> Spiking Network interface configuration:
    pops = ["mossy_fibers"]  # , "io_cell"]
    ports = [0]  # , 1]
    if config.NEST_PERIPHERY:
        pops = ['parrot_medulla', 'parrot_ponssens'] + pops
        ports = [0, 0] + ports
    for pop, receptor in zip(pops, ports):  #  excluding direct TVB input to "dcn_cell_glut_large"
        regions = neuron_types_to_region[pop]
        pop_regions_inds = []
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
        pop_regions_inds = np.array(pop_regions_inds)
        tvb_spikeNet_model_builder.output_interfaces.append(
            # !!! NOTE !!!
            # TVB cvoi is not the same with TVB-multiscale cvoi!!!
            # TVB simulator.model.cvoi denotes the state variables indices to be used
            # for the computation of large scale node coupling.
            # Instead, cosimulator.cosim_monitor.cvoi denotes the variable indices of the computed coupling,
            # to be used for sending data to some co-simulator, in our case, NEST.
            {'voi': np.array(["E"]),
             # # TVB thalamocortical model's coupling variables to get data from:
             # 0 for Isocortical and 1 for Subcortical (excluding specific thalami)
             "cvoi": np.array([0, 1]),
             'populations': np.array([pop]),  # NEST populations to couple to
              # --------------- Arguments that can default if not given by the user:------------------------------
              'model': 'RATE',  # This can be used to set default transformer and proxy models
              # 'coupling_mode': 'TVB',         # or "spikeNet", "NEST", etc
              'proxy_inds': pop_regions_inds,  # TVB proxy region nodes' indices
              # Set the enum entry or the corresponding label name for the "proxy_model",
              # or import and set the appropriate NEST proxy device class,
              # e.g., NESTInhomogeneousPoissonGeneratorSet, directly
              # options: "RATE", "RATE_TO_SPIKES", SPIKES", "PARROT_SPIKES" or CURRENT"
              # see tvb_multiscale.tvb_nest.interfaces.io.NESTInputProxyModels
             #  for options and related NESTDevice classes,
              # and tvb_multiscale.tvb_nest.interfaces.io.DefaultTVBtoNESTModels for the default choices
              'proxy_model': "RATE",
              'receptor_type': receptor,
              # Set the enum entry or the corresponding label name for the "transformer_model",
              # or import and set the appropriate tranformer class, e.g., ScaleRate, directly
              # options: "RATE", "SPIKES", "SPIKES_SINGE_INTERACTION", "SPIKES_MULTIPLE_INTERACTION", "CURRENT"
              # see tvb_multiscale.core.interfaces.transformers.models.DefaultTVBtoSpikeNetTransformers
              # for options and related Transformer classes,
              # and tvb_multiscale.core.interfaces.transformers.models.DefaultTVBtoSpikeNetModels
              # for default choices
              'transformer_model': "RATE",  # i.e., ThalamocorticalWCLinearRate,
             # Here the rate is a total rate, assuming a number of sending neurons:
             # Effective rate  = scale * (total_weighted_coupling_E_from_tvb - offset)
             # If E is in [0, 1.0], then, with a translation = 0.0, and a scale of 1e4
             # it is as if 100 neurons can fire each with a maximum spike rate of max_rate=100 Hz
              'transformer_params': {"scale_factor": np.array([config.w_TVB_to_NEST * simulator.model.G[0].item() * config.MOSSY_MAX_RATE])
                                     },   # "translation_factor": np.array([0.0])
              'spiking_proxy_inds': pop_regions_inds  # Same as "proxy_inds" for this kind of interface
              }
             )
    
    # These are user defined Spiking Network -> TVB interfaces configurations:
    pops = ["granule_cell", "dcn_cell_glut_large", "io_cell"]
    regs = [['Right Ansiform lobule', 'Left Ansiform lobule'],
            ['Right Cerebellar Nuclei', 'Left Cerebellar Nuclei'],
            ['Right Inferior olivary complex', 'Left Inferior olivary complex']]
    if config.NEST_PERIPHERY:
        pops = ["parrot_medulla", "parrot_ponssens"] + pops
        regs = [['Right Principal sensory nucleus of the trigeminal',
                 'Left Principal sensory nucleus of the trigeminal'],
                ['Right Pons Sensory', 'Left Pons Sensory']] + regs
    for iP, (pop, regions) in enumerate(zip(pops, regs)):
        pop_regions_inds = []
        numbers_of_neurons = nest_network.brain_regions[regions[0]][pop].number_of_neurons
        # Basic w to convert total rates to mean rates in Hz, and then into the interval [0.0, 1.0]:
        w_NEST_to_TVB = np.array([1.0]) / numbers_of_neurons / config.MOSSY_MAX_RATE
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
        pop_regions_inds = np.array(pop_regions_inds)
        if config.INVERSE_SIGMOIDAL_NEST_TO_TVB:
            # !!! Parameters for inverse sigmoidal NEST -> TVB transformer: !!!
            transformer_params = {"w": w_NEST_to_TVB,
                                  "Rmax": np.array([1.0 - 1e-9]),
                                  # We cannot allow 0 rate values leading the inverse sigmoidal to Inf!
                                  "Rmin": np.array([0.5*1e-4]),  # this leads to a minimum activity of -0.5 if...
                                  # ...we take beta and sigma value from model's sigmoidal activation function params:
                                  "beta": simulator.model.beta[[0]], "sigma": simulator.model.sigma[[0]]}
            w_name = "w"
        else:
            # !!! Parameters for linear NEST -> TVB transformer: !!!
            transformer_params = {"scale_factor": w_NEST_to_TVB, "translation_factor": np.array([-0.5])}
            w_name = "scale_factor"
        tvb_spikeNet_model_builder.input_interfaces.append(
            {'voi': np.array(['E']),
             'populations': np.array([pop]),
             'proxy_inds': pop_regions_inds,
             # --------------- Arguments that can default if not given by the user:------------------------------
             # Set the enum entry or the corresponding label name for the "proxy_model",
             # or import and set the appropriate NEST proxy device class, e.g., NESTSpikeRecorderMeanSet, directly
             # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
             # (the last two are identical for the moment returning all populations spikes together)
             # see tvb_multiscale.tvb_nest.interfaces.io.NESTOutputProxyModels for options and related NESTDevice classes,
             # and tvb_multiscale.tvb_nest.interfaces.io.DefaultNESTtoTVBModels for the default choices
             'proxy_model': "SPIKES_MEAN",
             # Set the enum entry or the corresponding label name for the "transformer_model",
             # or import and set the appropriate tranformer class, e.g., ElephantSpikesHistogramRate, directly
             # options: "SPIKES", "SPIKES_TO_RATE", "SPIKES_TO_HIST", "SPIKES_TO_HIST_RATE"
             # see tvb_multiscale.core.interfaces.transformers.models.DefaultSpikeNetToTVBTransformers for options and related Transformer classes,
             # and tvb_multiscale.core.interfaces.transformers.models.DefaultSpikeNetToTVBModels for default choices
             'transformer_model': "SPIKES_TO_HIST_RATE",
             'transformer_params': transformer_params
             })
        if pop == "granule_cell":
            # We only record from 1 every 10 granule cells!:
            tvb_spikeNet_model_builder.input_interfaces[-1]["neurons_fun"] = lambda i_node, nodes: nodes[0::10]
            tvb_spikeNet_model_builder.input_interfaces[-1]['transformer_params'][w_name] *= 10
    # Configure:
    tvb_spikeNet_model_builder.configure()
    # tvb_spikeNet_model_builder.print_summary_info_details(recursive=1)

    if config.VERBOSE > 1:
        # This is how the user defined TVB -> Spiking Network interface looks after configuration
        print("\noutput (TVB->NEST coupling) interfaces' configurations:\n")
        for interface in tvb_spikeNet_model_builder.output_interfaces:
            print(interface)

        # This is how the user defined Spiking Network -> TVB interfaces look after configuration
        print("\ninput (NEST->TVB update) interfaces' configurations:\n")
        for interface in tvb_spikeNet_model_builder.input_interfaces:
            print(interface)

    # Build the interfaces:
    simulator = tvb_spikeNet_model_builder.build()

    simulator.simulate_spiking_simulator = nest_network.nest_instance.Run  # set the method to run NEST

    if config.VERBOSE > 1:
        simulator.print_summary_info(recursive=3)
        # simulator.print_summary_info_details(recursive=3)

    if config.VERBOSE > 1:
        print("\n\noutput (TVB->NEST coupling) interfaces:\n")
        simulator.output_interfaces.print_summary_info_details(recursive=2)

        print("\n\ninput (NEST->TVB update) interfaces:\n")
        simulator.input_interfaces.print_summary_info_details(recursive=2)

    return simulator, nest_network


def simulate_tvb_nest(simulator, nest_network, config):
    simulator.simulation_length, transient = configure_simulation_length_with_transient(config)
    # Simulate and return results
    tic = time.time()
    if config.VERBOSE:
        print("Simulating TVB-NEST...")
    nest_network.nest_instance.Prepare()
    simulator.configure()
    # Adjust simulation length to be an integer multiple of synchronization_time:
    simulator.simulation_length = \
        np.ceil(simulator.simulation_length / simulator.synchronization_time) * simulator.synchronization_time
    results = simulator.run()
    nest_network.nest_instance.Run(nest_network.nest_instance.GetKernelStatus("resolution"))
    #  Cleanup NEST network unless you plan to continue simulation later
    nest_network.nest_instance.Cleanup()
    if config.VERBOSE:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    return results, transient, simulator, nest_network


def run_tvb_nest_workflow(PSD_target=None, model_params={}, config=None, write_files=True, **config_args):
    tic = time.time()
    from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import build_NEST_network, plot_nest_results_raster

    plot_flag = config_args.get('plot_flag', DEFAULT_ARGS.get('plot_flag'))
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    if config.VERBOSE:
        print("\n\n------------------------------------------------\n\n"+
              "Running TVB-NEST workflow for plot_flag=%s, write_files=%s,\nand model_params=\n%s...\n" 
              % (str(plot_flag), str(write_files), str(config.model_params)))
    # config.SIMULATION_LENGTH = 100.0
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Build NEST network
    nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
    # Build TVB-NEST interfaces
    simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
    # Simulate TVB-NEST model
    results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)
    results = tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
    if PSD_target is None:
        # This is the PSD target we are trying to fit...
        if config.model_params['G']:
            # ...for a connected brain, i.e., PS of bilateral M1 and S1:
            PSD_target = compute_target_PSDs_m1s1brl(config, write_files=True, plotter=plotter)
        else:
            # ...for a disconnected brain, average PS of all regions:
            PSD_target = compute_target_PSDs_1D(config, write_files=True, plotter=plotter)
    # This is the PSD computed from our simulation results...
    if config.model_params['G']:
        # ...for a connected brain, i.e., PS of bilateral M1 and S1:
        PSD = compute_data_PSDs_m1s1brl(results["source_ts"], PSD_target, inds, transient, plotter=plotter)
    else:
        # ...for a disconnected brain, average PS of all regions:
        PSD = compute_data_PSDs_1D(results["source_ts"], PSD_target, inds, transient, plotter=plotter)
    results.update({"PSD": PSD, "transient": transient,
                    "simulator": simulator, "nest_network": nest_network, "config": config})
    if plotter is not None:
        from examples.plot_write_results import plot_write_spiking_network_results
        results.update(plot_tvb(transient, inds, results,
                                simulator=simulator, plotter=plotter, config=config, write_files=write_files))
        plot_write_spiking_network_results(nest_network, connectivity=connectivity,
                                           time=results["source_ts"][0], transient=transient,
                                           monitor_period=simulator.monitors[0].period,
                                           plot_per_neuron=False, plotter=plotter, writer=None, config=config)
        plot_nest_results_raster(nest_network, neuron_models, neuron_number, config)
    if config.VERBOSE:
        print("\nFinished TVB-NEST workflow in %g sec!\n" % (time.time() - tic))
    return results


if __name__ == "__main__":
    parser = args_parser("tvb_nest_script")
    args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
    verbose = args.get('verbose', DEFAULT_ARGS['verbose'])
    if verbose:
        print("Running %s with arguments:\n" % parser.description)
        print(args, "\n")
    run_tvb_nest_workflow(**args)

# -*- coding: utf-8 -*-

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import neuron_types_to_region
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_script import *


def print_available_interfaces():
    # options for a nonopinionated builder:
    from tvb_multiscale.core.interfaces.base.transformers.models.models import Transformers
    from tvb_multiscale.core.interfaces.base.transformers.builders import \
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


def build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config):

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions

    #     # ---------------------------- Opinionated TVB<->NEST interface builder----------------------------
    #     from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import WilsonCowanTVBNESTInterfaceBuilder
    #     tvb_spikeNet_model_builder =  WilsonCowanTVBNESTInterfaceBuilder()  # opinionated builder

    # ---------------------------- Non opinionated TVB<->NEST interface builder----------------------------
    from tvb_multiscale.tvb_nest.interfaces.builders import TVBNESTInterfaceBuilder

    tvb_spikeNet_model_builder = TVBNESTInterfaceBuilder()  # non opinionated builder

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

    max_rate = 100.0  #Hz
    # if tvb_spikeNet_model_builder.default_coupling_mode == "TVB":
    #     proxy_inds = nest_nodes_inds
    # else:
    #     proxy_inds = np.arange(simulator.connectivity.number_of_regions).astype('i')
    #     proxy_inds = np.delete(proxy_inds, nest_nodes_inds)
    # This is a user defined TVB -> Spiking Network interface configuration:
    for pop, receptor in zip(['parrot_medulla', 'parrot_ponssens', "mossy_fibers", "io_cell"],
                             [0, 0, 0, 1]):  #  excluding direct TVB input to "dcn_cell_glut_large"
        regions = neuron_types_to_region[pop]
        pop_regions_inds = []
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
        pop_regions_inds = np.array(pop_regions_inds)
        tvb_spikeNet_model_builder.output_interfaces.append(
            {'voi': np.array(["E"]),  # TVB state variable to get data from
             'populations': np.array([pop]),  # NEST populations to couple to
              # --------------- Arguments that can default if not given by the user:------------------------------
              'model': 'RATE',  # This can be used to set default tranformer and proxy models
              # 'coupling_mode': 'TVB',         # or "spikeNet", "NEST", etc
              'proxy_inds': pop_regions_inds,  # TVB proxy region nodes' indices
              # Set the enum entry or the corresponding label name for the "proxy_model",
              # or import and set the appropriate NEST proxy device class, e.g., NESTInhomogeneousPoissonGeneratorSet, directly
              # options: "RATE", "RATE_TO_SPIKES", SPIKES", "PARROT_SPIKES" or CURRENT"
              # see tvb_multiscale.tvb_nest.interfaces.io.NESTInputProxyModels for options and related NESTDevice classes,
              # and tvb_multiscale.tvb_nest.interfaces.io.DefaultTVBtoNESTModels for the default choices
              'proxy_model': "RATE",
              'receptor_type': receptor,
              # Set the enum entry or the corresponding label name for the "transformer_model",
              # or import and set the appropriate tranformer class, e.g., ScaleRate, directly
              # options: "RATE", "SPIKES", "SPIKES_SINGE_INTERACTION", "SPIKES_MULTIPLE_INTERACTION", "CURRENT"
              # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetTransformers for options and related Transformer classes,
              # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetModels for default choices
              'transformer_model': "RATE",
             # Here the rate is a total rate, assuming a number of sending neurons:
             # Effective rate  = scale * (total_weighted_coupling_E_from_tvb - offset)
             # If E is in [0, 1.0], then, with a translation = 0.0, and a scale of 1e4
             # it is as if 100 neurons can fire each with a maximum spike rate of max_rate=100 Hz
              'transformer_params': {"scale_factor": np.array([1e2 * max_rate])},   # "translation_factor": np.array([0.0])
              'spiking_proxy_inds': pop_regions_inds  # Same as "proxy_inds" for this kind of interface
              }
             )

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for iP, (pop, sv, regions) \
            in enumerate(zip(["parrot_medulla", "parrot_ponssens", "granule_cell", "dcn_cell_glut_large", "io_cell"],
                             ["E", "E", "E", "E", "E"],
                             [['Right Principal sensory nucleus of the trigeminal',
                               'Left Principal sensory nucleus of the trigeminal'],
                              ['Right Pons Sensory', 'Left Pons Sensory'],
                              ['Right Ansiform lobule', 'Left Ansiform lobule'],
                              ['Right Interposed nucleus', 'Left Interposed nucleus'],
                              ['Right Inferior olivary complex', 'Left Inferior olivary complex']])):
        pop_regions_inds = []
        numbers_of_neurons = nest_network.brain_regions[regions[0]][pop].number_of_neurons
        time_scale_factor = 1e-3 * simulator.integrator.dt  # convert TVB time step to secs
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
        pop_regions_inds = np.array(pop_regions_inds)
        tvb_spikeNet_model_builder.input_interfaces.append(
            {'voi': np.array([sv]),
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
             # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBTransformers for options and related Transformer classes,
             # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBModels for default choices
             'transformer_model': "SPIKES_TO_HIST_RATE",
             'transformer_params': {"scale_factor": np.array([time_scale_factor]) / numbers_of_neurons / max_rate,
                                    "translation_factor": np.array([-0.5])}
             })
        if pop == "granule_cell":
            tvb_spikeNet_model_builder.input_interfaces[-1]["neurons_inds"] = lambda nodes: nodes[0::10]
            tvb_spikeNet_model_builder.input_interfaces[-1]['transformer_params']["scale_factor"] *= 10
    # Configure:
    tvb_spikeNet_model_builder.configure()
    # tvb_spikeNet_model_builder.print_summary_info_details(recursive=1)

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

    # simulator.print_summary_info(recursive=3)
    # simulator.print_summary_info_details(recursive=3)

    print("\n\noutput (TVB->NEST coupling) interfaces:\n")
    simulator.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (NEST->TVB update) interfaces:\n")
    simulator.input_interfaces.print_summary_info_details(recursive=2)

    return simulator, nest_network


def simulate_tvb_nest(simulator, nest_network, config,
                      neuron_models={}, neuron_number={}, print_flag=True, plot_flag=True):
    simulator.simulation_length, transient = configure_simulation_length_with_transient(config)
    # Simulate and return results
    tic = time.time()
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
    if print_flag:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    if plot_flag:
        from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import plot_nest_results
        plot_nest_results(nest_network, neuron_models, neuron_number, config)
    return results, transient, simulator, nest_network


def run_tvb_nest_workflow(G=5.0, STIMULUS=0.25,
                          I_E=-0.25, I_S=0.25,
                          W_IE=-3.0, W_RS=-2.0,
                          #TAU_E=10/0.9, TAU_I=10/0.9, TAU_S=10/0.25, TAU_R=10/0.25,
                          PSD_target=None, plot_flag=True, output_folder=None):

    from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import build_NEST_network

    # Get configuration
    config, plotter = configure(output_folder=output_folder, plot_flag=plot_flag)
    # config.SIMULATION_LENGTH = 100.0

    # Load connectome and other structural files
    connectome, major_structs_labels, voxel_count, inds = load_connectome(config, plotter=plotter)
    # Construct some more indices and maps
    inds, maps = construct_extra_inds_and_maps(connectome, inds)
    # Logprocess connectome
    connectome = logprocess_weights(connectome, inds, print_flag=True, plotter=plotter)
    # Prepare connectivity with all possible normalizations
    connectivity = build_connectivity(connectome, inds, config, print_flag=True, plotter=plotter)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, print_flag=True, plotter=plotter)
    # Build TVB-NEST interfaces
    nest_network, nest_nodes_inds, neuron_models, neuron_number, mossy_fibers_medulla, mossy_fibers_ponssens = \
        build_NEST_network(config)

    simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
    # Simulate TVB-NEST model
    results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config,
                                                                    neuron_models, neuron_number,
                                                                    plot_flag=True, print_flag=True)

    return results, transient, simulator, nest_network


if __name__ == "__main__":
    run_tvb_nest_workflow()
# -*- coding: utf-8 -*-

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import neuron_types_to_region


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
    tvb_spikeNet_model_builder.proxy_inds = nest_nodes_inds
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
    for pop in ['parrot', "mossy_fibers", "io_cell"]:  #  excluding direct TVB input to "dcn_cell_glut_large"
        regions = neuron_types_to_region[pop]
        pop_regions_inds = []
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
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
              # Set the enum entry or the corresponding label name for the "transformer_model",
              # or import and set the appropriate tranformer class, e.g., ScaleRate, directly
              # options: "RATE", "SPIKES", "SPIKES_SINGE_INTERACTION", "SPIKES_MULTIPLE_INTERACTION", "CURRENT"
              # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetTransformers for options and related Transformer classes,
              # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetModels for default choices
              'transformer_model': "RATE",
             # Here the rate is a total rate, assuming a number of sending neurons:
             # Effective rate  = scale * (total_weighted_coupling_E_from_tvb - offset)
             # If E is in [0, 1.0], then, with an offset = 0.0, and a scale of 1e4
             # it is as if 100 neurons can fire each with a maximum spike rate of max_rate=100 Hz
              'transformer_params': {"scale_factor": np.array([1e2 * max_rate])},   # "offset_factor": np.array([0.0])
              'spiking_proxy_inds': pop_regions_inds  # Same as "proxy_inds" for this kind of interface
              }
             )

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop, sv, regions in zip(["parrot", "parrot", "granule_cell", "dcn_cell_glut_large", "io_cell"],
                                ["E", "E", "E", "E", "E"],
                                [['Right Principal sensory nucleus of the trigeminal',
                                  'Left Principal sensory nucleus of the trigeminal'],
                                 ['Right Pons Sensory', 'Left Pons Sensory'],
                                 ['Right Ansiform lobule', 'Left Ansiform lobule'],
                                 ['Right Interposed nucleus', 'Left Interposed nucleus'],
                                ['Right Inferior olivary complex', 'Left Inferior olivary complex']]):
        pop_regions_inds = []
        numbers_of_neurons = nest_network.brain_regions[regions[0]][pop].number_of_neurons
        time_scale_factor = 1e-3 * simulator.integrator.dt  # convert TVB time step to secs
        for region in regions:
            pop_regions_inds.append(np.where(simulator.connectivity.region_labels == region)[0][0])
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
                                    "offset_factor": -0.5}
             }
        )

    # Configure:
    tvb_spikeNet_model_builder.configure()
    # tvb_spikeNet_model_builder.print_summary_info_details(recursive=1)

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (TVB->NEST coupling) interfaces' configurations:\n")
    print(tvb_spikeNet_model_builder.output_interfaces)

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (NEST->TVB update) interfaces' configurations:\n")
    print(tvb_spikeNet_model_builder.input_interfaces)

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

# -*- coding: utf-8 -*-

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.interfaces.base.transformers.models.models import Transformers
from tvb_multiscale.core.interfaces.base.transformers.builders import \
        DefaultTVBtoSpikeNetTransformers, DefaultSpikeNetToTVBTransformers, \
        DefaultTVBtoSpikeNetModels, DefaultSpikeNetToTVBModels
from tvb_multiscale.core.interfaces.base.builders import \
    TVBtoSpikeNetRemoteTransformerBuilder, SpikeNetToTVBRemoteTransformerBuilder


# FRONTEND used for user configuration of interfaces.

def print_enum(enum):
    print("\n", enum)
    for name, member in enum.__members__.items():
        print(name, "= ", member.value)


def available_models():
    # options for a nonopinionated builder:

    print("\n\nAvailable input (spikeNet->TVB update) / output (TVB->spikeNet coupling) transformer models:")

    print_enum(DefaultTVBtoSpikeNetModels)
    print_enum(DefaultTVBtoSpikeNetTransformers)

    print_enum(DefaultSpikeNetToTVBModels)
    print_enum(DefaultSpikeNetToTVBTransformers)

    print("\n\nAll basic transformer models:")
    print_enum(Transformers)


def prepare_TVBtoSpikeNet_transformer_interface(interface_builder=TVBtoSpikeNetRemoteTransformerBuilder,
                                                config_class=Config):
    config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons = configure(config_class)

    tvb_to_spikeNet_trans_interface_builder = None
    if np.all(SIM_MODE.lower() == "tvb-spikeNet"):
        tvb_to_spikeNet_trans_interface_builder = interface_builder(config=config)  # non opinionated builder

    if tvb_to_spikeNet_trans_interface_builder is not None:

        from tvb_multiscale.core.utils.file_utils import load_pickled_dict
        sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
        if not os.path.isfile(sim_serial_filepath):
            # In order to be independent create a TVB simulator, serialize it and write it to file:
            build_tvb_simulator();
        tvb_to_spikeNet_trans_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

        tvb_to_spikeNet_trans_interface_builder.input_label = "TVBtoTrans"
        tvb_to_spikeNet_trans_interface_builder.output_label = "TransToSpikeNet"
        # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
        # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
        # "1-to-1" TVB->spikeNet coupling.
        # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
        # large-scale coupling for spiking regions is computed in spikeNet,
        # taking into consideration the TVB connectome weights and delays, 
        # in this "1-to-many" TVB->spikeNet coupling.
        tvb_to_spikeNet_trans_interface_builder.proxy_inds = spiking_nodes_inds

        tvb_to_spikeNet_trans_interface_builder.output_interfaces = []
        tvb_to_spikeNet_trans_interface_builder.input_interfaces = []

    return tvb_to_spikeNet_trans_interface_builder


def prepare_spikeNetToTVB_transformer_interface(interfacec_builder=SpikeNetToTVBRemoteTransformerBuilder,
                                                config_class=Config):
    config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons = configure(config_class)

    spikeNet_to_tvb_trans_interface_builder = None
    if np.all(SIM_MODE.lower() == "tvb-spikeNet"):
        spikeNet_to_tvb_trans_interface_builder = interfacec_builder(config=config)  # non opinionated builder

    if spikeNet_to_tvb_trans_interface_builder is not None:

        from tvb_multiscale.core.utils.file_utils import load_pickled_dict
        sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
        if not os.path.isfile(sim_serial_filepath):
            # In order to be independent create a TVB simulator, serialize it and write it to file:
            build_tvb_simulator();
        spikeNet_to_tvb_trans_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

        spikeNet_to_tvb_trans_interface_builder.input_label = "spikeNetToTrans"
        spikeNet_to_tvb_trans_interface_builder.output_label = "TransToTVB"
        # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
        # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
        # "1-to-1" TVB->spikeNet coupling.
        # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
        # large-scale coupling for spiking regions is computed in spikeNet,
        # taking into consideration the TVB connectome weights and delays,
        # in this "1-to-many" TVB->spikeNet coupling.
        spikeNet_to_tvb_trans_interface_builder.proxy_inds = spiking_nodes_inds

        spikeNet_to_tvb_trans_interface_builder.output_interfaces = []
        spikeNet_to_tvb_trans_interface_builder.input_interfaces = []

    return spikeNet_to_tvb_trans_interface_builder


def build_TVBtoSpikeNet_transformer_interfaces(tvb_to_spikeNet_trans_interface_builder=None):
    if tvb_to_spikeNet_trans_interface_builder is None:
        tvb_to_spikeNet_trans_interface_builder = prepare_TVBtoSpikeNet_transformer_interface()

    # Load TVB to spikeNet interfaces configurations
    tvb_to_spikeNet_trans_interface_builder.load_all_interfaces()

    # Configure TVB to spikeNet interfaces' builder:
    tvb_to_spikeNet_trans_interface_builder.configure()
    # tvb_to_spikeNet_trans_interface_builder.print_summary_info_details(recursive=1)

    # Build TVB to spikeNet interfaces
    tvb_to_spikeNet_trans_interfaces = tvb_to_spikeNet_trans_interface_builder.build()

    print("\n\noutput (TVB -> ... -> Transformer -> ... -> spikeNet coupling) interfaces:\n")
    tvb_to_spikeNet_trans_interfaces.print_summary_info_details(recursive=2)

    return tvb_to_spikeNet_trans_interfaces


def build_spikeNetToTVB_transformer_interfaces(spikeNet_to_tvb_trans_interface_builder=None):
    if spikeNet_to_tvb_trans_interface_builder is None:
        spikeNet_to_tvb_trans_interface_builder = prepare_spikeNetToTVB_transformer_interface()

    # Load spikeNet to TVB interfaces configurations
    spikeNet_to_tvb_trans_interface_builder.load_all_interfaces()

    # Configure spikeNet to TVB interfaces' builder:
    spikeNet_to_tvb_trans_interface_builder.configure()
    # spikeNet_to_tvb_trans_interface_builder.print_summary_info_details(recursive=1)

    # Build spikeNet to TVB interfaces
    spikeNet_to_tvb_trans_interfaces = spikeNet_to_tvb_trans_interface_builder.build()

    print("\n\ninput (TVB<- ... <- Transformer <- ... <- spikeNet update) interfaces:\n")
    spikeNet_to_tvb_trans_interfaces.print_summary_info_details(recursive=2)

    return spikeNet_to_tvb_trans_interfaces

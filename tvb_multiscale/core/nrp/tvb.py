# -*- coding: utf-8 -*-

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import TVBRemoteInterfaceBuilder
from tvb_multiscale.core.nrp.config import configure


# This would run on TVB engine before creating any multiscale cosimulation interface connections.
# Users can adapt it to their use case.
def build_tvb_simulator(tvb_simulator_builder, config_class=Config):

    config = configure(config_class)[0]

    simulator = tvb_simulator_builder(config_class)

    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = serialize_tvb_cosimulator(simulator)
    print(sim_serial)

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    return simulator


# FRONTEND:
def prepare_TVB_interface(tvb_remote_interface_builder=TVBRemoteInterfaceBuilder, simulator=None, config_class=Config):
    config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons = configure(config_class)

    tvb_interface_builder = None
    if np.all(SIM_MODE.lower() == "tvb-spikeNet"):
        tvb_interface_builder = tvb_remote_interface_builder(config=config)  # non opinionated builder

    if tvb_interface_builder is not None:
        if simulator is not None:
            tvb_interface_builder.tvb_cosimulator = simulator
        tvb_interface_builder.input_label = "TransToTVB"
        tvb_interface_builder.output_label = "TVBtoTrans"
        tvb_interface_builder.proxy_inds = spiking_nodes_inds

        tvb_interface_builder.output_interfaces = []
        tvb_interface_builder.input_interfaces = []

    return tvb_interface_builder, spiking_nodes_inds


# BACKEND:
def build_TVB_interfaces(simulator, tvb_interface_builder=None):
    if tvb_interface_builder is None:
        tvb_interface_builder = prepare_TVB_interface(simulator=simulator)[0]
    else:
        tvb_interface_builder.tvb_cosimulator = simulator

    # Load TVB interfaces configurations
    tvb_interface_builder.load_all_interfaces()

    # Configure TVB interfaces' builder:
    tvb_interface_builder.configure()
    # tvb_interface_builder.print_summary_info_details(recursive=1)

    # Build interfaces and attach them to TVB simulator
    simulator = tvb_interface_builder.build()

    # simulator.print_summary_info(recursive=3)
    # simulator.print_summary_info_details(recursive=3)

    print("\n\noutput (TVB->spikeNet coupling) interfaces:\n")
    simulator.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (TVB<-spikNet update) interfaces:\n")
    simulator.input_interfaces.print_summary_info_details(recursive=2)

    return simulator


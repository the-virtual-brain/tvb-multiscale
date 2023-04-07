# -*- coding: utf-8 -*-

import os
import numpy as np

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.nrp.config import configure
from tvb_multiscale.core.nrp.tvb_config import build_tvb_simulator


# This would run on spiking simulator only before creating any multiscale cosimulation interface connections.
# Here it is assumed that the TVB simulator is already created and we can get some of its attributes,
# either by directly accessing it, or via serialization.
def build_spiking_network(spiking_simulator, spikeNet_builder, config_class=Config):

    config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons = configure(config_class)

    spiking_network = None

    if "spikeNet" in SIM_MODE.lower():

        # Load serialized TVB simulator from a file...:
        from tvb_multiscale.core.utils.file_utils import load_pickled_dict
        sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
        if not os.path.isfile(sim_serial_filepath):
            # In order to be independent create a TVB simulator, serialize it and write it to file:
            build_tvb_simulator(config_class);
        sim_serial = load_pickled_dict(sim_serial_filepath)

        if SPIKENET_MODEL_BUILDERS:

            if SPIKENET_MODEL_BUILDERS == "opinionated":

                # ------------------- Instantiating an opinionated spiking network builder for this model, -------------
                # using all default parameters for this example

                spikeNet_model_builder = spikeNet_builder(sim_serial,  # simulator,
                                                          spiking_nodes_inds=spiking_nodes_inds,
                                                          spiking_simulator=spiking_simulator, config=config)
                spikeNet_model_builder.population_order = n_neurons

            else:
                spikeNet_model_builder = \
                    spikeNet_builder(spiking_simulator, spiking_nodes_inds, n_neurons, sim_serial, config)

            spikeNet_model_builder.configure()

            spiking_network = spikeNet_model_builder.build()

        else:
            spiking_network = spikeNet_builder(spiking_simulator, spiking_nodes_inds, n_neurons, sim_serial, config)

        # Configure Spiking Network class:
        spiking_network.configure()
        spiking_network.print_summary_info_details(recursive=3, connectivity=True)

    else:
        spiking_nodes_inds = np.array([])

    return spiking_network, spiking_nodes_inds


# FRONTEND used for user configuration of interfaces.

def prepare_spikeNet_interface_builder(spikeNet_remote_interface_builder, config_class=Config):

    config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons = configure(config_class)

    spikeNet_interface_builder = None
    if np.all(SIM_MODE.lower() == "tvb-spikeNet"):

        # ---------------------------- Non opinionated TVB<->spikeNet interface builder----------------------------
        spikeNet_interface_builder = spikeNet_remote_interface_builder(config=config)  # non opinionated builder

        from tvb_multiscale.core.utils.file_utils import load_pickled_dict
        sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
        if not os.path.isfile(sim_serial_filepath):
            # In order to be independent create a TVB simulator, serialize it and write it to file:
            build_tvb_simulator();
        spikeNet_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

        spikeNet_interface_builder.input_label = "TransToSpikeNet"
        spikeNet_interface_builder.output_label = "spikeNetToTrans"

        spikeNet_interface_builder.proxy_inds = spiking_nodes_inds

        spikeNet_interface_builder.output_interfaces = []
        spikeNet_interface_builder.input_interfaces = []

    return spikeNet_interface_builder, spiking_nodes_inds, n_neurons


def build_spikeNet_interfaces(spiking_network, spikeNet_interface_builder=None):
    if spikeNet_interface_builder is None:
        spikeNet_interface_builder = prepare_spikeNet_interface_builder(spiking_network)[0]

    # Load spikeNet interfaces configurations
    spikeNet_interface_builder.load_all_interfaces()

    # Configure spikeNet interfaces' builder:
    spikeNet_interface_builder.configure()
    # spikeNet_interface_builder.print_summary_info_details(recursive=1)

    # Build spikeNet interfaces and attach them to spikeNet simulator
    spiking_network = spikeNet_interface_builder.build()

    print("\n\noutput (TVB->spikeNet coupling) interfaces:\n")
    spiking_network.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (TVB<-spikeNet update) interfaces:\n")
    spiking_network.input_interfaces.print_summary_info_details(recursive=2)

    return spiking_network

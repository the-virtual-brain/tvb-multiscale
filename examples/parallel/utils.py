# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb_multiscale.core.utils.file_utils import load_pickled_dict


def load_serial_tvb_cosimulator(config):
    sim_serial_path = getattr(config, "DEFAULT_TVB_SERIAL_COSIM_PATH",
                              os.path.join(config.FOLDER_CONFIG, "tvb_serial_cosimulator.pkl"))
    return load_pickled_dict(sim_serial_path)


def initialize_interface_builder_from_config(interface_builder, config):
    # This can be used to set default proxy or transformer models:
    interface_builder.model = config.INTERFACE_MODEL  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    interface_builder.proxy_inds = config.SPIKING_NODES_INDS
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
    # "1-to-1" TVB->spikeNet coupling.
    # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in spikeNet,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->spikeNet coupling.
    interface_builder.default_coupling_mode = config.INTERFACE_COUPLING_MODE  # "TVB" or "spikeNet"
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    interface_builder.exclusive_nodes = config.EXCLUSIVE_NODES
    return interface_builder


def get_default_interface_builder_tvb_proxy_inds(interface_builder, spiking_nodes_inds, number_of_brain_regions):
    if interface_builder.default_coupling_mode == "TVB":
        proxy_inds = interface_builder.proxy_inds
    else:
        proxy_inds = np.arange(number_of_brain_regions).astype('i')
        proxy_inds = np.delete(proxy_inds, spiking_nodes_inds)
    return proxy_inds

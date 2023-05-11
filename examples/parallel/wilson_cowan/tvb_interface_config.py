# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder, TVBRemoteInterfaceBuilder

from examples.parallel.wilson_cowan.config import configure


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:


def configure_TVB_interfaces(simulator=None, tvb_interface_builder_class=TVBInterfaceBuilder,
                             config=None, config_class=Config, dump_configs=True):

    if config is None:
        config = configure(config_class)

    # Configuring a nonopinionated builder:
    tvb_interface_builder = tvb_interface_builder_class(config=config)

    if simulator is not None:
        tvb_interface_builder.tvb_cosimulator = simulator

    # This can be used to set default tranformer and proxy models:
    tvb_interface_builder.model = config.INTERFACE_MODEL  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
    # "1-to-1" TVB->spikeNet coupling.
    # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in spikeNet,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->spikeNet coupling.
    tvb_interface_builder.default_coupling_mode = config.INTERFACE_COUPLING_MODE  # "TVB" or "spikeNet"
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    tvb_interface_builder.exclusive_nodes = config.EXCLUSIVE_NODES

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_interface_builder.output_interfaces = \
        [{'voi': np.array(["E"]),  # TVB state variable to get data from
          # --------------- Arguments that can default if not given by the user:------------------------------
          'model': 'RATE',  # This can be used to set default tranformer and proxy models
          'coupling_mode': 'TVB',  # or "spikeNet", "spikeNet", etc
          'proxy_inds': config.spiking_nodes_inds  # TVB proxy region nodes' indices
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop, sv in zip(["E", "I"], ["E", "I"]):
        tvb_interface_builder.input_interfaces.append(
            {'voi': np.array([sv]),
             'proxy_inds': config.spiking_nodes_inds
             }
        )

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (TVB-> coupling) interfaces' configurations:\n")

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (TVB<- update) interfaces' configurations:\n")

    if dump_configs:
        tvb_interface_builder.dump_all_interfaces()

    return tvb_interface_builder


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:
def configure_TVB_remote_interfaces(simulator=None, tvb_interface_class=TVBRemoteInterfaceBuilder,
                                    config=None, config_class=Config):
    return configure_TVB_interfaces(simulator, tvb_interface_class, config, config_class, True)

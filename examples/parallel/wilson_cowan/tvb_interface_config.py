# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder, TVBRemoteInterfaceBuilder

from examples.parallel.utils import \
    initialize_interface_builder_from_config, get_default_interface_builder_tvb_proxy_inds
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
    tvb_interface_builder = initialize_interface_builder_from_config(tvb_interface_builder, config)
    tvb_proxy_inds = get_default_interface_builder_tvb_proxy_inds(tvb_interface_builder,
                                                                  config.SPIKING_NODES_INDS,
                                                                  simulator.connectivity.number_of_regions)

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_interface_builder.output_interfaces = \
        [{'voi': np.array(["E"]),  # TVB state variable to get data from
          # --------------- Arguments that can default if not given by the user:------------------------------
          'coupling_mode': config.INTERFACE_COUPLING_MODE,  # or "spikeNet", "spikeNet", etc
          'proxy_inds': tvb_proxy_inds  # TVB proxy region nodes' indices
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    tvb_interface_builder.input_interfaces = []
    for pop, sv in zip(["E", "I"], ["E", "I"]):
        tvb_interface_builder.input_interfaces.append(
            {'voi': np.array([sv]),
             'proxy_inds': tvb_interface_builder.proxy_inds
             }
        )

    if dump_configs:
        tvb_interface_builder.dump_all_interfaces()

    if config.VERBOSITY:
        # This is how the user defined TVB -> Spiking Network interface looks after configuration
        print("\noutput (TVB-> coupling) interfaces' configurations:\n")
        print(tvb_interface_builder.output_interfaces)

        # This is how the user defined Spiking Network -> TVB interfaces look after configuration
        print("\ninput (TVB<- update) interfaces' configurations:\n")
        print(tvb_interface_builder.input_interfaces)

    return tvb_interface_builder


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:
def configure_TVB_remote_interfaces(simulator=None, tvb_interface_class=TVBRemoteInterfaceBuilder,
                                    config=None, config_class=Config):
    return configure_TVB_interfaces(simulator, tvb_interface_class, config, config_class, True)

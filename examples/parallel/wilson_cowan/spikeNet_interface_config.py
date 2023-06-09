# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb_multiscale.core.config import Config

from examples.parallel.utils import load_serial_tvb_cosimulator, \
    initialize_interface_builder_from_config, get_default_interface_builder_tvb_proxy_inds
from examples.parallel.wilson_cowan.config import configure


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:
def configure_spikeNet_interfaces(spike_interface_builder_class,
                                  config=None, config_class=Config, dump_configs=True):

    if config is None:
        config = configure(config_class)

    spikeNet_interface_builder = spike_interface_builder_class(config=config)
    spikeNet_interface_builder.tvb_simulator_serialized = load_serial_tvb_cosimulator(config)
    spikeNet_interface_builder = initialize_interface_builder_from_config(spikeNet_interface_builder, config)
    spikeNet_interface_builder.N_E = config.N_NEURONS
    spikeNet_interface_builder.N_I = config.N_NEURONS
    tvb_proxy_inds = \
        get_default_interface_builder_tvb_proxy_inds(
            spikeNet_interface_builder,
            config.SPIKING_NODES_INDS,
            spikeNet_interface_builder.tvb_simulator_serialized["connectivity.number_of_regions"])

    # This is a user defined TVB -> Spiking Network interface configuration:
    spikeNet_interface_builder.input_interfaces = \
        [{'populations': np.array(["E"]),  # spikeNet populations to couple to
          # --------------- Arguments that can default if not given by the user:------------------------------
          # 'model': config.INTERFACE_MODEL,  # This can be used to set default transformer and proxy models
          'coupling_mode': config.INTERFACE_COUPLING_MODE,  # or "spikeNet", "spikeNet", etc
          'proxy_inds': tvb_proxy_inds,  # TVB proxy region nodes' indices
          # Set the enum entry or the corresponding label name for the "proxy_model",
          # or import and set the appropriate spikeNet proxy device class directly
          # options: "RATE", "RATE_TO_SPIKES", SPIKES", "PARROT_SPIKES" or CURRENT"
          'proxy_model': config.TVB_TO_SPIKENET_PROXY_MODEL,
          'spiking_proxy_inds': spikeNet_interface_builder.proxy_inds
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    spikeNet_interface_builder.output_interfaces = []
    for pop in ["E", "I"]:
        spikeNet_interface_builder.output_interfaces.append(
            {'populations': np.array([pop]),
             'proxy_inds': spikeNet_interface_builder.proxy_inds,
             # --------------- Arguments that can default if not given by the user:------------------------------
             # Set the enum entry or the corresponding label name for the "proxy_model",
             # or import and set the appropriate spikeNet proxy device class directly
             # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
             # (the last two are identical for the moment returning all populations spikes together)
             'proxy_model': "SPIKES_MEAN"
             }
        )

    if config.VERBOSITY:
        # This is how the user defined Spiking Network -> TVB interfaces look after configuration
        print("\ninput (spikeNet <- coupling) interfaces' configurations:\n")
        print(spikeNet_interface_builder.input_interfaces)

        # This is how the user defined TVB -> Spiking Network interface looks after configuration
        print("\noutput (spikeNet -> update) interfaces' configurations:\n")
        print(spikeNet_interface_builder.output_interfaces)

    if dump_configs:
        spikeNet_interface_builder.dump_all_interfaces()

    return spikeNet_interface_builder

# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.nrp.spikeNet import prepare_spikeNet_interface_builder


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:
def configure_spikeNet_interfaces():

    spikeNet_interface_builder, spiking_nodes_inds, n_neurons = prepare_spikeNet_interface_builder(spiking_network)

    #     # Using all default parameters for this example of an opinionated builder
    #     tvb_spikeNet_model_builder.default_config()

    # or setting a nonopinionated builder:

    # This can be used to set default tranformer and proxy models:
    spikeNet_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
    # "1-to-1" TVB->spikeNet coupling.
    # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in spikeNet,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->spikeNet coupling.
    spikeNet_interface_builder.default_coupling_mode = "TVB"
    # Number of neurons per population to be used to compute population mean instantaneous firing rates:
    spikeNet_interface_builder.N_E = n_neurons
    spikeNet_interface_builder.N_I = n_neurons
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    spikeNet_interface_builder.exclusive_nodes = True
    
    # This is a user defined TVB -> Spiking Network interface configuration:
    spikeNet_interface_builder.input_interfaces = \
        [{'populations': np.array(["E"]),  # spikeNet populations to couple to
          # --------------- Arguments that can default if not given by the user:------------------------------
          'model': 'RATE',  # This can be used to set default tranformer and proxy models
          'coupling_mode': 'TVB',  # or "spikeNet", "spikeNet", etc
          'proxy_inds': spiking_nodes_inds,  # TVB proxy region nodes' indices
          # Set the enum entry or the corresponding label name for the "proxy_model",
          # or import and set the appropriate spikeNet proxy device class directly
          # options: "RATE", "RATE_TO_SPIKES", SPIKES", "PARROT_SPIKES" or CURRENT"
          'proxy_model': "RATE",
          'spiking_proxy_inds': spiking_nodes_inds  # Same as "proxy_inds" for this kind of interface
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop in ["E", "I"]:
        spikeNet_interface_builder.output_interfaces.append(
            {'populations': np.array([pop]),
             'proxy_inds': spiking_nodes_inds,
             # --------------- Arguments that can default if not given by the user:------------------------------
             # Set the enum entry or the corresponding label name for the "proxy_model",
             # or import and set the appropriate spikeNet proxy device class directly
             # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
             # (the last two are identical for the moment returning all populations spikes together)
             'proxy_model': "SPIKES_MEAN",
             }
        )

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (spikeNet -> coupling) interfaces' configurations:\n")
    display(spikeNet_interface_builder.output_interfaces)

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (spikeNet <- update) interfaces' configurations:\n")
    display(spikeNet_interface_builder.input_interfaces)

    spikeNet_interface_builder.dump_all_interfaces()

    return spikeNet_interface_builder

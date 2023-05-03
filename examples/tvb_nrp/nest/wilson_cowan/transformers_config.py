# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.tvb_nest.config import Config
from examples.tvb_nrp.nest.wilson_cowan.config import configure


# FRONTEND used for user configuration of interfaces.

def configure_TVBtoSpikeNet_transformer_interfaces(config=None, config_class=Config):

    from tvb_multiscale.core.interfaces.base.builders import TVBtoSpikeNetTransformerBuilder
    from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels

    if config is None:
        from examples.tvb_nrp.nest.wilson_cowan.config import configure
        config = configure(config_class)

    tvb_to_spikeNet_trans_interface_builder = TVBtoSpikeNetTransformerBuilder(config=config)

    # This can be used to set default tranformer and proxy models:
    tvb_to_spikeNet_trans_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    tvb_to_spikeNet_trans_interface_builder.N_E = config.n_neurons
    tvb_to_spikeNet_trans_interface_builder.N_I = config.n_neurons

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_to_spikeNet_trans_interface_builder.output_interfaces = \
        [{  # Set the enum entry or the corresponding label name for the "transformer_model", 
            # or import and set the appropriate transformer class, e.g., ScaleRate, directly
            # options: "RATE", "SPIKES", "SPIKES_SINGE_INTERACTION", "SPIKES_MULTIPLE_INTERACTION", "CURRENT"
            # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetTransformers
            # for options and related Transformer classes,
            # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetModels for default choices
            'transformer_model': "RATE"
        }
        ]

    for interface in tvb_to_spikeNet_trans_interface_builder.output_interfaces:
        # The "scale_factor" scales the TVB state variable to convert it to an 
        # instantaneous rate:
        if tvb_to_spikeNet_trans_interface_builder.model == TVBtoSpikeNetModels.SPIKES.name:
            # The "number_of_neurons" will determine how many spike trains will be generated:
            interface["transformer_params"] = \
                {"scale_factor": np.array([100]),
                 "number_of_neurons": np.array([tvb_to_spikeNet_trans_interface_builder.N_E])}
        else:  # RATE
            # Here the rate is a total rate, assuming a number of sending neurons:
            interface["transformer_params"] = {"scale_factor":
                                                   1e6 * np.array([tvb_to_spikeNet_trans_interface_builder.N_E])}

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (->Transformer-> coupling) interfaces' configurations:\n")

    tvb_to_spikeNet_trans_interface_builder.dump_all_interfaces()

    return tvb_to_spikeNet_trans_interface_builder


def configure_spikeNetToTVB_transformer_interfaces(config=None, config_class=Config):

    from tvb_multiscale.core.interfaces.base.builders import SpikeNetToTVBTransformerBuilder
    # from tvb_multiscale.core.interfaces.tvb.interfaces import SpikeNetToTVBModels

    if config is None:
        from examples.tvb_nrp.nest.wilson_cowan.config import configure
        config = configure(config_class)

    spikeNet_to_TVB_transformer_interface_builder = SpikeNetToTVBTransformerBuilder(config=config)

    # This can be used to set default transformer and proxy models:
    spikeNet_to_TVB_transformer_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    spikeNet_to_TVB_transformer_interface_builder.N_E = config.n_neurons
    spikeNet_to_TVB_transformer_interface_builder.N_I = config.n_neurons

    for ii, N in enumerate([spikeNet_to_TVB_transformer_interface_builder.N_E,
                            spikeNet_to_TVB_transformer_interface_builder.N_I]):
        spikeNet_to_TVB_transformer_interface_builder.input_interfaces.append(
            {  # Set the enum entry or the corresponding label name for the "transformer_model", 
                # or import and set the appropriate tranformer class, e.g., ElephantSpikesHistogramRate, directly
                # options: "SPIKES", "SPIKES_TO_RATE", "SPIKES_TO_HIST", "SPIKES_TO_HIST_RATE"
                # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBTransformers for options and related Transformer classes,
                # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBModels for default choices
                "transformer_model": "SPIKES_TO_HIST_RATE",
                # The "scale_factor" scales the instantaneous rate coming from spikeNet, before setting it to TVB,
                # in our case converting the rate to a mean reate 
                # and scaling it to be in the TVB model's state variable range [0.0, 1.0]
                "transformer_params": {"scale_factor": np.array([1e-4]) / N}
            })

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (TVB<-...-Transformer<-...-spikeNet update) interfaces' configurations:\n")

    spikeNet_to_TVB_transformer_interface_builder.dump_all_interfaces()

    return spikeNet_to_TVB_transformer_interface_builder

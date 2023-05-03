# -*- coding: utf-8 -*-

from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder
from tvb_multiscale.core.orchestrators.nrp_apps import NRPTransformerApp


def transformers_init(config, transformer_interfaces_builder):

    # Create a NRPTVBApp
    transformer_app = NRPTransformerApp(config=config,
                                        proxy_inds=config.PROXY_INDS,
                                        synchronization_time=getattr(config, "SYNCHRONIZATION_TIME", 0.0),
                                        interfaces_builder=transformer_interfaces_builder,
                                        simulation_length=config.SIMULATION_LENGTH)

    transformer_app.start()
    # Configure App (and CoSimulator and interface builders)
    transformer_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    transformer_app.build()

    return transformer_app


def tvb_to_spikeNet_transformer_init(config, transformer_interfaces_builder=TVBtoSpikeNetTransformerBuilder):

    return transformers_init(config, transformer_interfaces_builder)


def spikeNet_to_tvb_transformer_init(config, transformer_interfaces_builder=SpikeNetToTVBTransformerBuilder):
    return transformers_init(config, transformer_interfaces_builder)

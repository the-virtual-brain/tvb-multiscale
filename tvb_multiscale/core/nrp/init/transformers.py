# -*- coding: utf-8 -*-

from tvb_multiscale.core.orchestrators.nrp_apps import NRPTransformerApp, \
    NRPTVBtoSpikeNetTransformerApp, NRPSpikeNetToTVBTransformerApp


def transformers_init(config, transformer_app_class=NRPTransformerApp, **kwargs):

    # Create a TransformerApp
    transformer_app = transformer_app_class(config=config, **kwargs)

    transformer_app.start()
    # Configure App (and Transformer interface builders)
    transformer_app.configure()

    # Build (Transformer interfaces)
    transformer_app.build()

    # Configure App for CoSimulation
    transformer_app.configure_simulation()

    return transformer_app


def tvb_to_spikeNet_transformer_init(config, **kwargs):
    return transformers_init(config, NRPTVBtoSpikeNetTransformerApp, **kwargs)


def spikeNet_to_tvb_transformer_init(config, **kwargs):
    return transformers_init(config, NRPSpikeNetToTVBTransformerApp, **kwargs)

# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.orchestrators.nrp_apps import NRPSpikeNetApp


def spikeNet_init(config, nrp_spikenet_app=NRPSpikeNetApp, spiking_network_builder=None, **kwargs):

    # Create a NRPTVBApp
    spikeNet_app = nrp_spikenet_app(config=config,
                                    proxy_inds=config.PROXY_INDS,
                                    synchronization_time=getattr(config.SYNCHRONIZATION_TIME, 0.0),
                                    exclusive_nodes=getattr(config.EXCLUSIVE_NODES, True),
                                    simulation_length=config.SIMULATION_LENGTH,
                                    **kwargs)

    # Set...
    if isinstance(spiking_network_builder, SpikingNetworkBuilder):
        # ...a Spiking Network builder class instance:
        spikeNet_app.spiking_network_builder = spiking_network_builder
    else:
        # ...or, assuming a callable function script, build and set the Spiking Network
        spikeNet_app.spiking_network = spiking_network_builder(config)

    spikeNet_app.start()
    # Configure App (and CoSimulator and interface builders)
    spikeNet_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    spikeNet_app.build()

    return spikeNet_app

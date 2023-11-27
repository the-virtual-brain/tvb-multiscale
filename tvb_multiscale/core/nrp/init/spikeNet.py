# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.orchestrators.nrp_apps import NRPSpikeNetApp


def spikeNet_init(config, nrp_spikenet_app=NRPSpikeNetApp, spiking_network_builder=None, **kwargs):

    # Create a NRPTVBApp
    spikeNet_app = nrp_spikenet_app(config=config,
                                    synchronization_time=getattr(config, "SYNCHRONIZATION_TIME", 0.0),
                                    **kwargs)

    # Set...
    if isinstance(spiking_network_builder, SpikingNetworkBuilder):
        # ...a Spiking Network builder class instance:
        spikeNet_app.spikeNet_builder = spiking_network_builder
    else:
        # ...or, a callable function
        spikeNet_app.spikeNet_builder_function = spiking_network_builder

    spikeNet_app.start()
    # Configure App (and CoSimulator and interface builders)
    spikeNet_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    spikeNet_app.build()

    # Configure App for CoSimulation
    spikeNet_app.configure_simulation()

    return spikeNet_app


def nest_init(config, spiking_network_builder=None, **kwargs):
    from tvb_multiscale.tvb_nest.orchestrators import NESTNRPApp
    return spikeNet_init(config, nrp_spikenet_app=NESTNRPApp,
                         spiking_network_builder=spiking_network_builder, **kwargs)

def annarchy_init(config, spiking_network_builder=None, **kwargs):
    from tvb_multiscale.tvb_annarchy.orchestrators import ANNarchyNRPApp
    return spikeNet_init(config, nrp_spikenet_app=ANNarchyNRPApp,
                         spiking_network_builder=spiking_network_builder, **kwargs)

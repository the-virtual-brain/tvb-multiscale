# -*- coding: utf-8 -*-

from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorNRPBuilder
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder
from tvb_multiscale.core.orchestrators.nrp_apps import NRPTVBApp


def tvb_init(config, tvb_cosimulator_builder=None, tvb_interfaces_builder=TVBInterfaceBuilder):

    # Create a NRPTVBApp
    tvb_app = NRPTVBApp(config=config,
                        proxy_inds=config.PROXY_INDS,
                        synchronization_time=getattr(config.SYNCHRONIZATION_TIME, 0.0),
                        exclusive_nodes=getattr(config.EXCLUSIVE_NODES, True),
                        interfaces_builder=tvb_interfaces_builder,
                        simulation_length=config.SIMULATION_LENGTH)

    # Set...
    if isinstance(tvb_cosimulator_builder, CoSimulatorNRPBuilder):
        # ...a TVB CoSimulator builder class instance:
        tvb_app.cosimulator_builder = tvb_cosimulator_builder
    else:
        # ...or, assuming a callable function script, build and set the TVB CoSimulator
        tvb_app.cosimulator = tvb_cosimulator_builder(config)

    tvb_app.start()
    # Configure App (and CoSimulator and interface builders)
    tvb_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    tvb_app.build()

    return tvb_app

# -*- coding: utf-8 -*-

from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorParallelBuilder
from tvb_multiscale.core.orchestrators.nrp_apps import NRPTVBApp


def tvb_init(config, tvb_cosimulator_builder=None, **kwargs):

    # Create a TVBApp
    tvb_app = NRPTVBApp(config=config, **kwargs)
    # Set...
    if isinstance(tvb_cosimulator_builder, CoSimulatorParallelBuilder):
        # ...a TVB CoSimulator builder class instance:
        tvb_app.cosimulator_builder = tvb_cosimulator_builder
    else:
        # ...or, a callable function
        tvb_app.cosimulator_builder_function = tvb_cosimulator_builder

    tvb_app.start()
    # Configure App (and CoSimulator and interface builders)
    tvb_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    tvb_app.build()

    # Configure App for CoSimulation
    tvb_app.configure_simulation()

    return tvb_app

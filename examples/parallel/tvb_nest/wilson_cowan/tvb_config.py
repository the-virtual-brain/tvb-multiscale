# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import Config

from examples.parallel.wilson_cowan.tvb_config import build_tvb_simulator as build_tvb_simulator_base


# This would run on TVB process before creating any multiscale cosimulation interface connections.
# Users can adapt it to their use case.
def build_tvb_simulator(config=None, config_class=Config, **kwargs):
    return build_tvb_simulator_base(config, config_class, **kwargs)

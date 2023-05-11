# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import Config

from examples.parallel.wilson_cowan.tvb_interface_config import \
    configure_TVB_interfaces as configure_TVB_interfaces_base


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:
def configure_TVB_interfaces(simulator=None, config=None, config_class=Config, dump_configs=True):
    return configure_TVB_interfaces_base(simulator, config, config_class, dump_configs)

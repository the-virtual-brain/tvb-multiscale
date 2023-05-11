# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import Config

from examples.parallel.wilson_cowan.tvb_interface_config import \
    configure_TVB_interfaces as configure_TVB_interfaces_base, \
    configure_TVB_remote_interfaces as configure_TVB_remote_interfaces_base


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:


def configure_TVB_interfaces(simulator=None, config=None, config_class=Config, dump_configs=True, **kwargs):
    return configure_TVB_interfaces_base(simulator=simulator,
                                         config=config, config_class=config_class, dump_configs=dump_configs, **kwargs)


def configure_TVB_remote_interfaces(simulator=None, config=None, config_class=Config, **kwargs):
    return configure_TVB_remote_interfaces_base(simulator=simulator,
                                                config=config, config_class=config_class, **kwargs)

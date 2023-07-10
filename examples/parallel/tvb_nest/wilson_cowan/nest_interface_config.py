# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import Config
from tvb_multiscale.tvb_nest.interfaces.builders import NESTInterfaceBuilder, NESTRemoteInterfaceBuilder

from examples.parallel.wilson_cowan.spikeNet_interface_config import \
    configure_spikeNet_interfaces as configure_spikeNet_interfaces_base


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:


def configure_NEST_interfaces(spike_interface_builder_class=NESTInterfaceBuilder,
                              config=None, config_class=Config, dump_configs=True):
    return configure_spikeNet_interfaces_base(spike_interface_builder_class,
                                              config, config_class, dump_configs)


def configure_NEST_remote_interfaces(spike_interface_builder_class=NESTRemoteInterfaceBuilder,
                                     config=None, config_class=Config):
    return configure_spikeNet_interfaces_base(spike_interface_builder_class,
                                              config, config_class)

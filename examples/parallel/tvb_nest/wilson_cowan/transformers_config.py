# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import Config

from examples.parallel.wilson_cowan.transformers_config import \
    configure_TVBtoSpikeNet_transformer_interfaces, configure_spikeNetToTVB_transformer_interfaces, \
    configure_TVBtoSpikeNet_remote_transformer_interfaces, configure_spikeNetToTVB_remote_transformer_interfaces


# FRONTEND used for user configuration of interfaces.
# These is an example that could be modified by users:


def configure_TVBtoNEST_transformer_interfaces(config=None, config_class=Config, dump_configs=True, **kwargs):
    return configure_TVBtoSpikeNet_transformer_interfaces(config=config, config_class=config_class,
                                                          dump_configs=dump_configs, **kwargs)


def configure_TVBtoNEST_remote_transformer_interfaces(config=None, config_class=Config, **kwargs):
    return configure_TVBtoSpikeNet_remote_transformer_interfaces(config=config, config_class=config_class, **kwargs)


def configure_NESTtoTVB_transformer_interfaces(config=None, config_class=Config, dump_configs=True, **kwargs):
    return configure_spikeNetToTVB_transformer_interfaces(config=config, config_class=config_class,
                                                          dump_configs=dump_configs, **kwargs)


def configure_NESTtoTVB_remote_transformer_interfaces(config=None, config_class=Config, **kwargs):
    return configure_spikeNetToTVB_remote_transformer_interfaces(config=config, config_class=config_class, **kwargs)

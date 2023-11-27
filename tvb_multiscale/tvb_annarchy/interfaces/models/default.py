# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import \
    ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, TVBANNarchyInterfaceBuilder


class DefaultANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultANNarchyInterfaceBuilder(DefaultANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder,
                                      DefaultSpikeNetInterfaceBuilder):
    pass


class DefaultTVBANNarchyInterfaceBuilder(DefaultANNarchyProxyNodesBuilder, TVBANNarchyInterfaceBuilder,
                                         DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)

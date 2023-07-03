# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.wilson_cowan import WilsonCowanTVBSpikeNetInterfaceBuilder, \
    WilsonCowanSpikeNetRemoteInterfaceBuilder, WilsonCowanSpikeNetTransformerInterfaceBuilder, \
    WilsonCowanSpikeNetInterfaceBuilder, WilsonCowanSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, \
    ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder, \
    ANNarchyTransformerInterfaceBuilder


class WilsonCowanANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder, WilsonCowanSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanANNarchyInterfaceBuilder(WilsonCowanANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder,
                                      WilsonCowanSpikeNetInterfaceBuilder):
    pass


class WilsonCowanANNarchyRemoteInterfaceBuilder(WilsonCowanANNarchyInterfaceBuilder, ANNarchyRemoteInterfaceBuilder,
                                            WilsonCowanSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class WilsonCowanANNarchyTransformerInterfaceBuilder(WilsonCowanANNarchyInterfaceBuilder,
                                                     ANNarchyTransformerInterfaceBuilder,
                                                     WilsonCowanSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class WilsonCowanTVBANNarchyInterfaceBuilder(WilsonCowanANNarchyProxyNodesBuilder,
                                             TVBANNarchyInterfaceBuilder,
                                             WilsonCowanTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_input_config(self)

# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.wilson_cowan import WilsonCowanTVBSpikeNetInterfaceBuilder, \
    WilsonCowanSpikeNetRemoteInterfaceBuilder, WilsonCowanSpikeNetTransformerInterfaceBuilder, \
    WilsonCowanSpikeNetOutputTransformerInterfaceBuilder, WilsonCowanSpikeNetInputTransformerInterfaceBuilder, \
    WilsonCowanSpikeNetInterfaceBuilder, WilsonCowanSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, \
    ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder, \
    ANNarchyTransformerInterfaceBuilder, ANNarchyOutputTransformerInterfaceBuilder, \
    ANNarchyInputTransformerInterfaceBuilder


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


class WilsonCowanNESNetOutputTransformerInterfaceBuilder(WilsonCowanANNarchyInterfaceBuilder,
                                                         ANNarchyOutputTransformerInterfaceBuilder,
                                                         WilsonCowanSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class WilsonCowanANNarchyInputTransformerInterfaceBuilder(WilsonCowanANNarchyInterfaceBuilder,
                                                      ANNarchyInputTransformerInterfaceBuilder,
                                                      WilsonCowanSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


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

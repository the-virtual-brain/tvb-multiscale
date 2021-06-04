# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.cereb import \
    CerebTVBSpikeNetInterfaceBuilder, \
    CerebSpikeNetRemoteInterfaceBuilder, CerebSpikeNetTransformerInterfaceBuilder, \
    CerebSpikeNetOutputTransformerInterfaceBuilder, \
    CerebSpikeNetInputTransformerInterfaceBuilder, \
    CerebSpikeNetInterfaceBuilder, CerebSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, TVBNESTInterfaceBuilder, \
    NESTTransformerInterfaceBuilder, NESTOutputTransformerInterfaceBuilder, NESTInputTransformerInterfaceBuilder


class CerebNESTProxyNodesBuilder(NESTProxyNodesBuilder, CerebSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class CerebNESTInterfaceBuilder(CerebNESTProxyNodesBuilder, NESTInterfaceBuilder, CerebSpikeNetInterfaceBuilder):

    pass


class CerebNESTRemoteInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTRemoteInterfaceBuilder, CerebSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        CerebSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class CerebNESTOutputTransformerInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTOutputTransformerInterfaceBuilder, CerebSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        CerebSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class CerebNESTInputTransformerInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTInputTransformerInterfaceBuilder, CerebSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        CerebSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class CerebNESTTransformerInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTTransformerInterfaceBuilder, CerebSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        CerebSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class CerebTVBNESTInterfaceBuilder(
    CerebNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,  CerebTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        CerebTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebTVBSpikeNetInterfaceBuilder.default_input_config(self)

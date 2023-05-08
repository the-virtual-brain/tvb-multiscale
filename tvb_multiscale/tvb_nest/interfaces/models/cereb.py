# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.cereb import \
    CerebTVBSpikeNetInterfaceBuilder, \
    CerebSpikeNetRemoteInterfaceBuilder, CerebSpikeNetTransformerInterfaceBuilder, \
    CerebSpikeNetInterfaceBuilder, CerebSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, NESTTransformerInterfaceBuilder, TVBNESTInterfaceBuilder


class CerebNESTProxyNodesBuilder(NESTProxyNodesBuilder, CerebSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class CerebNESTInterfaceBuilder(CerebNESTProxyNodesBuilder, NESTInterfaceBuilder, CerebSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


class CerebNESTRemoteInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTRemoteInterfaceBuilder, CerebSpikeNetRemoteInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        CerebSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class CerebNESTTransformerInterfaceBuilder(
    CerebNESTInterfaceBuilder, NESTTransformerInterfaceBuilder, CerebSpikeNetTransformerInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        CerebSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class CerebTVBNESTInterfaceBuilder(
    CerebNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,  CerebTVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        CerebTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        CerebTVBSpikeNetInterfaceBuilder.default_input_config(self)

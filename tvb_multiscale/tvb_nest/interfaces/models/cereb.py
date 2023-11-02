# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.cereb import \
    CerebTVBSpikeNetInterfaceBuilder, CerebSpikeNetInterfaceBuilder, CerebSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import \
    NESTProxyNodesBuilder, NESTInterfaceBuilder, TVBNESTInterfaceBuilder


class CerebNESTProxyNodesBuilder(NESTProxyNodesBuilder, CerebSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class CerebNESTInterfaceBuilder(CerebNESTProxyNodesBuilder, NESTInterfaceBuilder, CerebSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


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

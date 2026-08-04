# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, TVBNESTInterfaceBuilder

from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class DefaultNESTProxyNodesBuilder(NESTProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultNESTInterfaceBuilder(DefaultNESTProxyNodesBuilder, NESTInterfaceBuilder, DefaultSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


class DefaultTVBNESTInterfaceBuilder(DefaultNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                     DefaultTVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseNESTProxyNodesBuilder(DefaultNESTProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    def _default_receptor_type(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)


class DefaultMultisynapseNESTInterfaceBuilder(DefaultMultisynapseNESTProxyNodesBuilder, DefaultNESTInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


class DefaultMultisynapseTVBNESTInterfaceBuilder(DefaultMultisynapseNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                                 DefaultTVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)

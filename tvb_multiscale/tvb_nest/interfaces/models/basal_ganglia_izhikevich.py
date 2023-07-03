# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder, BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, NESTTransformerInterfaceBuilder, TVBNESTInterfaceBuilder


class BasalGangliaIzhikevichNESTProxyNodesBuilder(NESTProxyNodesBuilder,
                                                  BasalGangliaIzhikevichSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class BasalGangliaIzhikevichNESTInterfaceBuilder(BasalGangliaIzhikevichNESTProxyNodesBuilder, NESTInterfaceBuilder, 
                                                 BasalGangliaIzhikevichSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


class BasalGangliaIzhikevichNESTRemoteInterfaceBuilder(BasalGangliaIzhikevichNESTInterfaceBuilder,
                                                       NESTRemoteInterfaceBuilder,
                                                       BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class BasalGangliaIzhikevichNESTTransformerInterfaceBuilder(
    BasalGangliaIzhikevichNESTInterfaceBuilder, NESTTransformerInterfaceBuilder,
    BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class BasalGangliaIzhikevichTVBNESTInterfaceBuilder(
    BasalGangliaIzhikevichNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_input_config(self)

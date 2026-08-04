# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import \
    NESTProxyNodesBuilder, NESTInterfaceBuilder, TVBNESTInterfaceBuilder


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

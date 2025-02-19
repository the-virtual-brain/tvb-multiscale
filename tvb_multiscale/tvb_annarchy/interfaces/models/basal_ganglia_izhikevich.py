# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import \
    ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, TVBANNarchyInterfaceBuilder, TVBtoANNarchyModels


class BasalGangliaIzhikevichANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder,
                                                      BasalGangliaIzhikevichSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    def _default_receptor_type(self, source_node, target_node):
        if self.model == TVBtoANNarchyModels.CURRENT.name:
            return "exc"
        else:
            return "ampa"


class BasalGangliaIzhikevichANNarchyInterfaceBuilder(BasalGangliaIzhikevichANNarchyProxyNodesBuilder,
                                                     ANNarchyInterfaceBuilder,
                                                     BasalGangliaIzhikevichSpikeNetInterfaceBuilder):

    pass


class BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder(
    BasalGangliaIzhikevichANNarchyProxyNodesBuilder, TVBANNarchyInterfaceBuilder,
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_input_config(self)

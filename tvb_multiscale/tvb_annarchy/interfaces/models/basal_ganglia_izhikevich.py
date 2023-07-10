# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder, BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, \
    ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder, \
    ANNarchyTransformerInterfaceBuilder, TVBtoANNarchyModels


class BasalGangliaIzhikevichANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder,
                                                      BasalGangliaIzhikevichSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    __default_receptor_type = "ampa"

    def _default_receptor_type(self, source_node, target_node):
        if self.model == TVBtoANNarchyModels.CURRENT.name:
            return "exc"
        else:
            return "ampa"


class BasalGangliaIzhikevichANNarchyInterfaceBuilder(BasalGangliaIzhikevichANNarchyProxyNodesBuilder,
                                                     ANNarchyInterfaceBuilder,
                                                     BasalGangliaIzhikevichSpikeNetInterfaceBuilder):

    pass


class BasalGangliaIzhikevichANNarchyRemoteInterfaceBuilder(BasalGangliaIzhikevichANNarchyInterfaceBuilder,
                                                           ANNarchyRemoteInterfaceBuilder,
                                                           BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class BasalGangliaIzhikevichANNarchyTransformerInterfaceBuilder(
    BasalGangliaIzhikevichANNarchyInterfaceBuilder, ANNarchyTransformerInterfaceBuilder,
    BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder(
    BasalGangliaIzhikevichANNarchyProxyNodesBuilder, TVBANNarchyInterfaceBuilder,
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder.default_input_config(self)

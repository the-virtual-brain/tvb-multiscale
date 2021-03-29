# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder, BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetOutputTransformerInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInputTransformerInterfaceBuilder, \
    BasalGangliaIzhikevichSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, \
    ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder, \
    ANNarchyTransformerInterfaceBuilder, ANNarchyOutputTransformerInterfaceBuilder, \
    ANNarchyInputTransformerInterfaceBuilder


class BasalGangliaIzhikevichANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder,
                                                  BasalGangliaIzhikevichSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


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


class BasalGangliaIzhikevichANNarchyOutputTransformerInterfaceBuilder(
    BasalGangliaIzhikevichANNarchyInterfaceBuilder, ANNarchyOutputTransformerInterfaceBuilder,
    BasalGangliaIzhikevichSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class BasalGangliaIzhikevichANNarchyInputTransformerInterfaceBuilder(
    BasalGangliaIzhikevichANNarchyInterfaceBuilder, ANNarchyInputTransformerInterfaceBuilder,
    BasalGangliaIzhikevichSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


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

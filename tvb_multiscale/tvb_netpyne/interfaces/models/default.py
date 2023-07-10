from abc import ABCMeta, ABC

import numpy as np

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetRemoteInterfaceBuilder, DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels
from tvb_multiscale.core.interfaces.models.red_wong_wang import RedWongWangExcIOInhITVBInterfaceBuilder, \
    RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder, RedWongWangExcIOInhISpikeNetProxyNodesBuilder
from tvb_multiscale.tvb_netpyne.interfaces.builders import NetpyneProxyNodesBuilder, NetpyneInterfaceBuilder, \
    NetpyneRemoteInterfaceBuilder, TVBNetpyneInterfaceBuilder


class DefaultNetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultNetpyneInterfaceBuilder(DefaultNetpyneProxyNodesBuilder, NetpyneInterfaceBuilder, DefaultSpikeNetInterfaceBuilder):
    pass


class DefaultTVBNetpyneInterfaceBuilder(DefaultNetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder,
                                        DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder,
                                                   RedWongWangExcIOInhISpikeNetProxyNodesBuilder):

    def _default_receptor_type(self, source_node, target_node):
        return None


class RedWongWangExcIOInhITVBNetpyneInterfaceBuilder(RedWongWangExcIOInhINetpyneProxyNodesBuilder,
                                                     TVBNetpyneInterfaceBuilder,
                                                     RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhITVBInterfaceBuilder.default_output_config(self)

        transformer_params = {}
        if self.model == TVBtoSpikeNetModels.RATE.name:
            # due to the way Netpyne generates spikes, no scaling by population size is needed
            transformer_params = {"scale_factor": np.array([1.0])}

        self.output_interfaces[0]["transformer_params"] = transformer_params
        self.output_interfaces[0]["populations"] = "E"
        self.output_interfaces[0]["proxy_params"] = {"number_of_neurons": self.N_E}
        if self.synaptic_model_funcs:
            self.output_interfaces[0]["receptor_type"] = self.synaptic_model_funcs()['E']

        if self.lamda > 0.0:
            from copy import deepcopy
            self.output_interfaces[1]["transformer_params"] = deepcopy(transformer_params)
            self.output_interfaces[1]["populations"] = "I"
            self.output_interfaces[1]["proxy_params"] = {"number_of_neurons": self.N_I,
                                                         "lamda": self.lamda}
            if self.synaptic_model_funcs:
                self.output_interfaces[1]["receptor_type"] = self.synaptic_model_funcs()['I']

    def default_input_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_input_config(self)
          
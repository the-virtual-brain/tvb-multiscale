from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetRemoteInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder
    
from tvb_multiscale.core.interfaces.models.red_wong_wang import RedWongWangExcIOInhISpikeNetInterfaceBuilder

from tvb_multiscale.tvb_netpyne.interfaces.builders import NetpyneProxyNodesBuilder, NetpyneInterfaceBuilder, \
    NetpyneRemoteInterfaceBuilder, TVBNetpyneInterfaceBuilder

class DefaultNetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultNetpyneInterfaceBuilder(DefaultNetpyneProxyNodesBuilder, NetpyneInterfaceBuilder, DefaultSpikeNetInterfaceBuilder):
    pass


class DefaultNetpyneRemoteInterfaceBuilder(DefaultNetpyneInterfaceBuilder, NetpyneRemoteInterfaceBuilder,
                                        DefaultSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_input_config(self)

class DefaultTVBNetpyneInterfaceBuilder(DefaultNetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder,
                                     DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)

from tvb_multiscale.core.interfaces.models.red_wong_wang import \
    RedWongWangExcIOTVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOSpikeNetRemoteInterfaceBuilder, RedWongWangExcIOSpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetOutputTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetInputTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetInterfaceBuilder, RedWongWangExcIOSpikeNetProxyNodesBuilder, \
    RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder, RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetOutputTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetInputTransformerInterfaceBuilder, \
    RedWongWangExcIOInhITVBInterfaceBuilder, RedWongWangExcIOInhISpikeNetProxyNodesBuilder

class RedWongWangExcIOInhINetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder,
                                                RedWongWangExcIOInhISpikeNetProxyNodesBuilder):

    def _default_receptor_type(self, source_node, target_node):
        return "exc" # TODO: de-hardcode

class RedWongWangExcIOInhITVBNetpyneInterfaceBuilder(RedWongWangExcIOInhINetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder,
                                                     RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_input_config(self)
          
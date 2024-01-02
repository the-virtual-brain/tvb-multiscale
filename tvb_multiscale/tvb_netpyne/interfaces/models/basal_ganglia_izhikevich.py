from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder, BasalGangliaIzhikevichSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_netpyne.interfaces.builders import NetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder

class BasalGangliaIzhikevichNetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder,
                                                      BasalGangliaIzhikevichSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    def _default_receptor_type(self, source_node, target_node):
        return None

class BasalGangliaIzhikevichTVBNetpyneInterfaceBuilder(
    BasalGangliaIzhikevichNetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder,
    BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder):

    pass
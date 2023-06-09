# -*- coding: utf-8 -*-

from abc import ABCMeta

from tvb.basic.neotraits.api import HasTraits, Attr, List

from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBReceiverInterfaces, TVBtoSpikeNetModels, \
    SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetOutputTransformerInterface, SpikeNetInputTransformerInterface, \
    SpikeNetSenderInterface, SpikeNetReceiverInterface, \
    SpikeNetTransformerSenderInterface, SpikeNetReceiverTransformerInterface, \
    SpikeNetOutputInterfaces, SpikeNetInputInterfaces, \
    SpikeNetOutputTransformerInterfaces, SpikeNetInputTransformerInterfaces, \
    SpikeNetSenderInterfaces, SpikeNetReceiverInterfaces, \
    SpikeNetTransformerSenderInterfaces, SpikeNetReceiverTransformerInterfaces

from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.interfaces.io import \
    NESTInputDeviceSet, NESTOutputDeviceSet


TVBtoNESTModels = TVBtoSpikeNetModels
NESTtoTVBModels = SpikeNetToTVBModels


class NESTInterface(HasTraits):
    __metaclass__ = ABCMeta

    """NESTInterface base class for interfaces sending/receiving data from/to NEST."""

    spiking_network = Attr(label="NEST Network",
                           doc="""The instance of NESTNetwork class""",
                           field_type=NESTNetwork,
                           required=True)

    @property
    def nest_network(self):
        return self.spiking_network

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance


class NESTOutputInterface(NESTInterface, SpikeNetOutputInterface):

    """NESTOutputInterface base class for interfaces sending data from NEST."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NESTOutputDeviceSet implementing a proxy node 
                        sending outputs from the NEST network to the co-simulator""",
                 field_type=NESTOutputDeviceSet,
                 required=True)

    @property
    def _time(self):
        return self.nest_instance.GetKernelStatus("biological_time")

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)


class NESTOutputTransformerInterface(NESTOutputInterface, SpikeNetOutputTransformerInterface):

    """NESTOutputTransformerInterface"""

    pass


class NESTSenderInterface(NESTOutputInterface, SpikeNetSenderInterface):

    """NESTSenderInterface"""

    pass


class NESTTransformerSenderInterface(NESTOutputInterface, SpikeNetTransformerSenderInterface):
    """NESTTransformerSenderInterface"""

    pass


class NESTInputInterface(NESTInterface, SpikeNetInputInterface):

    """NESTInputInterface base class for interfaces receiving data to NEST."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NESTInputDeviceSet implementing a proxy node 
                        receiving inputs from the co-simulator as an input to the NEST network""",
                 field_type=NESTInputDeviceSet,
                 required=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)


class NESTInputTransformerInterface(NESTInputInterface, SpikeNetInputTransformerInterface):

    """NESTInputTransformerInterface"""

    pass


class NESTReceiverInterface(NESTInputInterface, SpikeNetReceiverInterface):
    """NESTReceiverInterface"""

    pass


class NESTReceiverTransformerInterface(NESTInputInterface, SpikeNetReceiverTransformerInterface):
    """NESTReceiverTransformerInterface"""

    pass


class TVBtoNESTInterface(NESTInputInterface, TVBtoSpikeNetInterface):

    """TVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memmory.
    """

    pass


class NESTtoTVBInterface(NESTOutputInterface, SpikeNetToTVBInterface):

    """NESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    pass

    pass


class NESTInterfaces(HasTraits):

    """NESTInterfaces class holding a list of NESTInterface instances"""

    interfaces = List(of=NESTInterface)

    @property
    def proxy_gids(self):
        return self._loop_get_from_interfaces("proxy_gids")

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    @property
    def nest_network(self):
        return self.spiking_network

    @property
    def nest_instance(self):
        if len(self.interfaces):
            return self.interfaces[0].nest_instance
        else:
            return None


class NESTOutputInterfaces(SpikeNetOutputInterfaces, NESTInterfaces):

    """NESTSenderInterfaces holding a list of NESTSenderInterface instances"""

    interfaces = List(of=NESTOutputInterface)


class NESTInputInterfaces(SpikeNetInputInterfaces, NESTInterfaces):

    """NESTInputInterfaces holding a list of NESTInputInterface instances"""

    interfaces = List(of=NESTInputInterface)


class NESTOutputTransformerInterfaces(SpikeNetOutputTransformerInterfaces, NESTInterfaces):

    """NESTOutputTransformerInterfaces holding a list of NESTOutputTransformerInterface instances"""

    interfaces = List(of=NESTOutputTransformerInterface)


class NESTInputTransformerInterfaces(SpikeNetInputTransformerInterfaces, NESTInterfaces):

    """NESTInputTransformerInterfaces holding a list of NESTInputTransformerInterface instances"""

    interfaces = List(of=NESTInputTransformerInterface)


class NESTSenderInterfaces(SpikeNetSenderInterfaces, NESTInterfaces):

    """NESTSenderInterfaces holding a list of NESTSenderInterface instances"""

    interfaces = List(of=NESTSenderInterface)


class NESTReceiverInterfaces(SpikeNetReceiverInterfaces, NESTInterfaces):

    """NESTReceiverInterfaces holding a list of NESTReceiverInterface instances"""

    interfaces = List(of=NESTReceiverInterface)


class NESTTransformerSenderInterfaces(SpikeNetTransformerSenderInterfaces, NESTInterfaces):

    """NESTTransformerSenderInterfaces holding a list of NESTTransformerSenderInterface instances"""

    interfaces = List(of=NESTTransformerSenderInterface)


class NESTReceiverTransformerInterfaces(SpikeNetReceiverTransformerInterfaces, NESTInterfaces):

    """NESTReceiverTransformerInterfaces holding a list of NESTReceiverTransformerInterface instances"""

    interfaces = List(of=NESTReceiverTransformerInterface)


class TVBtoNESTInterfaces(TVBOutputInterfaces, NESTInputInterfaces):

    """TVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    interfaces = List(of=TVBtoNESTInterface)


class NESTtoTVBInterfaces(TVBReceiverInterfaces, NESTOutputInterfaces):
    """NESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    interfaces = List(of=NESTtoTVBInterface)

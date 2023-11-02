# -*- coding: utf-8 -*-

from enum import Enum
from abc import ABCMeta

from tvb.basic.neotraits.api import HasTraits, Attr, List

from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBReceiverInterfaces, TVBtoSpikeNetModels, \
    SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetSenderInterface, SpikeNetReceiverInterface

from tvb_multiscale.tvb_annarchy.interfaces.io import ANNarchyInputDeviceSet, ANNarchyOutputDeviceSet
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork


# TVBtoANNarchyModels = TVBtoSpikeNetModels
class TVBtoANNarchyModels(Enum):
    RATE = 0
    SPIKES = 1


ANNarchytoTVBModels = SpikeNetToTVBModels


class ANNarchyInterface(HasTraits):
    __metaclass__ = ABCMeta

    """ANNarchyInterface base class for interfaces sending/receiving data from/to ANNarchy."""

    spiking_network = Attr(label="ANNarchy Network",
                           doc="""The instance of ANNarchyNetwork class""",
                           field_type=ANNarchyNetwork,
                           required=True)

    @property
    def annarchy_network(self):
        return self.spiking_network

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance


class ANNarchyOutputInterface(ANNarchyInterface, SpikeNetOutputInterface):

    """ANNarchyOutputInterface base class for interfaces sending data from ANNarchy."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of ANNarchyOutputDeviceSet implementing a proxy node 
                        sending outputs from the ANNarchy network to the co-simulator""",
                 field_type=ANNarchyOutputDeviceSet,
                 required=True)

    @property
    def _time(self):
        return self.annarchy_instance.get_time()

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)


class ANNarchySenderInterface(ANNarchyOutputInterface, SpikeNetSenderInterface):

    """ANNarchySenderInterface"""

    pass


class ANNarchyInputInterface(ANNarchyInterface, SpikeNetInputInterface):

    """ANNarchyInputInterface base class for interfaces receiving data to ANNarchy."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of ANNarchyInputDeviceSet implementing a proxy node 
                        receiving inputs from the co-simulator as an input to the ANNarchy network""",
                 field_type=ANNarchyInputDeviceSet,
                 required=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)


class ANNarchyReceiverInterface(ANNarchyInputInterface, SpikeNetReceiverInterface):

    """ANNarchyReceiverInterface"""

    pass


class TVBtoANNarchyInterface(ANNarchyInputInterface, TVBtoSpikeNetInterface):

    """TVBtoANNarchyInterface class to get data from TVB, transform them,
       and finally set them to ANNarchy, all processes taking place in shared memory.
    """

    pass


class ANNarchyToTVBInterface(ANNarchyOutputInterface, SpikeNetToTVBInterface):

    """ANNarchyToTVBInterface class to get data from ANNarchy, transform them,
       and finally set them to TVB, all processes taking place in shared memory.
    """

    pass


class ANNarchyInterfaces(HasTraits):

    """ANNarchyInterfaces class holding a list of ANNarchyInterface instances"""

    interfaces = List(of=ANNarchyInterface)

    @property
    def proxy_gids(self):
        return self._loop_get_from_interfaces("proxy_gids")

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    @property
    def annarchy_network(self):
        return self.spiking_network

    @property
    def annarchy_instance(self):
        if len(self.interfaces):
            return self.interfaces[0].annarchy_instance
        else:
            return None


class ANNarchyOutputInterfaces(SpikeNetOutputInterfaces, ANNarchyInterfaces):

    """ANNarchySenderInterfaces holding a list of ANNarchySenderInterface instances"""

    interfaces = List(of=ANNarchyOutputInterface)


class ANNarchyInputInterfaces(SpikeNetInputInterfaces, ANNarchyInterfaces):

    """ANNarchyInputInterfaces holding a list of ANNarchyInputInterface instances"""

    interfaces = List(of=ANNarchyInputInterface)



class ANNarchySenderInterfaces(SpikeNetSenderInterfaces, ANNarchyInterfaces):

    """ANNarchySenderInterfaces holding a list of ANNarchySenderInterface instances"""

    interfaces = List(of=ANNarchySenderInterface)


class ANNarchyReceiverInterfaces(SpikeNetReceiverInterfaces, ANNarchyInterfaces):

    """ANNarchyReceiverInterfaces holding a list of ANNarchyReceiverInterface instances"""

    interfaces = List(of=ANNarchyReceiverInterface)


class TVBtoANNarchyInterfaces(TVBOutputInterfaces, ANNarchyInputInterfaces):

    """TVBtoANNarchyInterfaces class holding a list of TVBtoANNarchyInterface instances"""

    interfaces = List(of=TVBtoANNarchyInterface)


class ANNarchyToTVBInterfaces(TVBReceiverInterfaces, ANNarchyOutputInterfaces):

    """ANNarchyToTVBInterfaces class holding a list of ANNarchyToTVBInterface instances"""

    interfaces = List(of=ANNarchyToTVBInterface)

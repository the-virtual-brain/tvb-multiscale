# -*- coding: utf-8 -*-

from enum import Enum
from abc import ABCMeta

from tvb.basic.neotraits.api import HasTraits, Attr, List

from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBInputInterfaces, TVBtoSpikeNetModels, \
    SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetOutputRemoteInterfaces, SpikeNetInputRemoteInterfaces,\
    SpikeNetSenderInterface, SpikeNetReceiverInterface, \
    SpikeNetTransformerSenderInterface, SpikeNetReceiverTransformerInterface

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

    def __call__(self):
        return self.send(ANNarchyOutputInterface.get_proxy_data(self))


class ANNarchyTransformerSenderInterface(ANNarchyOutputInterface, SpikeNetTransformerSenderInterface):
    
    """ANNarchyTransformerSenderInterface"""

    def __call__(self):
        return self.transform_send(ANNarchyOutputInterface.get_proxy_data(self))


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


class ANNarchyReceiverTransformerInterface(ANNarchyInputInterface, SpikeNetReceiverTransformerInterface):

    """ANNarchyReceiverTransformerInterface"""

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

    def get_proxy_data(self):
        return ANNarchyOutputInterface.get_proxy_data(self)


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
        if len(self.interfaces):
            return self.interfaces[0].spiking_network
        else:
            return None

    @property
    def annarchy_instance(self):
        if len(self.interfaces):
            return self.interfaces[0].annarchy_instance
        else:
            return None


class ANNarchyOutputInterfaces(ANNarchyInterfaces, SpikeNetOutputRemoteInterfaces):

    """ANNarchyOutputInterfaces holding a list of ANNarchyOutputInterface instances"""

    pass


class ANNarchyInputInterfaces(ANNarchyInterfaces, SpikeNetInputRemoteInterfaces):

    """ANNarchyInputInterfaces holding a list of ANNarchyInputInterface instances"""

    pass


class TVBtoANNarchyInterfaces(ANNarchyInputInterfaces, TVBOutputInterfaces):

    """TVBtoANNarchyInterfaces class holding a list of TVBtoANNarchyInterface instances"""

    pass


class ANNarchyToTVBInterfaces(ANNarchyOutputInterfaces, TVBInputInterfaces):

    """ANNarchyToTVBInterfaces class holding a list of ANNarchyToTVBInterface instances"""

    pass

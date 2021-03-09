# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits.api import HasTraits, Attr, List
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBInputInterfaces
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputRemoteInterfaces, SpikeNetInputRemoteInterfaces,\
    SpikeNetSenderInterface, SpikeNetReceiverInterface, \
    SpikeNetTransformerSenderInterface, SpikeNetReceiverTransformerInterface, \
    TVBtoSpikeNetModels, SpikeNetToTVBModels

from tvb_multiscale.tvb_nest.interfaces.io import \
    NESTInputDeviceSet, NESTOutputDeviceSet
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork


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

    @property
    @abstractmethod
    def proxy_gids(self):
        pass

    def _get_proxy_gids(self, nest_devices):
        gids = nest_devices.get("global_id")
        if isinstance(gids, dict):
            return gids["global_id"]

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    def print_str(self):
        return "\nNEST proxy nodes' gids:\n%s" % extract_integer_intervals(self.proxy_gids, print=True)


class NESTOutputInterface(NESTInterface):

    """NESTOutputInterface base class for interfaces sending data from NEST."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NESTOutputDeviceSet implementing a proxy node 
                        sending outputs from the NEST network to the co-simulator""",
                 field_type=NESTOutputDeviceSet,
                 required=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)


class NESTSenderInterface(SpikeNetSenderInterface, NESTOutputInterface):

    """NESTSenderInterface"""

    def print_str(self):
        SpikeNetSenderInterface.print_str(self) + NESTOutputInterface.print_str(self)


class NESTTransformerSenderInterface(SpikeNetTransformerSenderInterface, NESTOutputInterface):
    """NESTTransformerSenderInterface"""

    def print_str(self):
        SpikeNetTransformerSenderInterface.print_str(self) + NESTOutputInterface.print_str(self)


class NESTInputInterface(NESTInterface):

    """NESTInputInterface base class for interfaces receiving data to NEST."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NESTInputDeviceSet implementing a proxy node 
                        receiving inputs from the co-simulator as an input to the NEST network""",
                 field_type=NESTInputDeviceSet,
                 required=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)


class NESTReceiverInterface(SpikeNetReceiverInterface, NESTInputInterface):
    """NESTReceiverInterface"""

    def print_str(self):
        SpikeNetReceiverInterface.print_str(self) + NESTInputInterface.print_str(self)


class NESTReceiverTransformerInterface(SpikeNetReceiverTransformerInterface, NESTInputInterface):
    """NESTReceiverTransformerInterface"""

    def print_str(self):
        SpikeNetReceiverTransformerInterface.print_str(self) + NESTInputInterface.print_str(self)


class TVBtoNESTInterface(TVBtoSpikeNetInterface, NESTInputInterface):

    """TVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memmory.
    """

    def print_str(self):
        TVBtoSpikeNetInterface.print_str(self) + NESTInputInterface.print_str(self)


class NESTtoTVBInterface(SpikeNetToTVBInterface, NESTOutputInterface):

    """NESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    def print_str(self):
        SpikeNetToTVBInterface.print_str(self) + NESTOutputInterface.print_str(self)


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
        if len(self.interfaces):
            return self.interfaces[0].spiking_network
        else:
            return None

    @property
    def nest_instance(self):
        if len(self.interfaces):
            return self.interfaces[0].nest_instance
        else:
            return None


class NESTOutputInterfaces(SpikeNetOutputRemoteInterfaces, NESTInterfaces):

    """NESTOutputInterfaces holding a list of NESTOutputInterface instances"""

    pass


class NESTInputInterfaces(SpikeNetInputRemoteInterfaces, NESTInterfaces):

    """NESTInputInterfaces holding a list of NESTInputInterface instances"""

    pass


class TVBtoNESTInterfaces(TVBOutputInterfaces, NESTInputInterfaces):

    """TVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    pass


class NESTtoTVBInterfaces(TVBInputInterfaces, NESTOutputInterfaces):
    """NESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    pass

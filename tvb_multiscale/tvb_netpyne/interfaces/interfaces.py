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

from tvb_multiscale.tvb_netpyne.interfaces.io import \
    NetpyneInputDeviceSet, NetpyneOutputDeviceSet
from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork


TVBtoNetpyneModels = TVBtoSpikeNetModels
NetpyneToTVBModels = SpikeNetToTVBModels


class NetpyneInterface(HasTraits):
    __metaclass__ = ABCMeta

    """NetpyneInterface base class for interfaces sending/receiving data from/to NetPyNE."""

    spiking_network = Attr(label="NetPyNE Network",
                           doc="""The instance of NetpyneNetwork class""",
                           field_type=NetpyneNetwork,
                           required=True)

    @property
    def netpyne_network(self):
        return self.spiking_network

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

    @property
    def time(self):
        return self.netpyne_instance.time


class NetpyneOutputInterface(NetpyneInterface, SpikeNetOutputInterface):

    """NetpyneOutputInterface base class for interfaces sending data from NetPyNE."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NetpyneOutputDeviceSet implementing a proxy node 
                        sending outputs from the NetPyNE network to the co-simulator""",
                 field_type=NetpyneOutputDeviceSet,
                 required=True)

    @property
    def _time(self):
        return self.netpyne_instance.time

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)


class NetpyneInputInterface(NetpyneInterface, SpikeNetInputInterface):

    """NetpyneInputInterface base class for interfaces receiving data to NetPyNE."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of NetpyneInputDeviceSet implementing a proxy node 
                        receiving inputs from the co-simulator as an input to the NetPyNE network""",
                 field_type=NetpyneInputDeviceSet,
                 required=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)


class TVBtoNetpyneInterface(NetpyneInputInterface, TVBtoSpikeNetInterface):

    """TVBtoNetpyneInterface class to get data from TVB, transform them,
       and finally set them to NetPyNE, all processes taking place in shared memmory.
    """

    pass


class NetpyneToTVBInterface(NetpyneOutputInterface, SpikeNetToTVBInterface):

    """NetpyneToTVBInterface class to get data from NetPyNE, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    def get_proxy_data(self):
        return NetpyneOutputInterface.get_proxy_data(self)
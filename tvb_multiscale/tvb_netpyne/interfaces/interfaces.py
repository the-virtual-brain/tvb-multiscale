from abc import ABCMeta

from tvb.basic.neotraits.api import HasTraits, Attr, List

from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBReceiverInterfaces, TVBtoSpikeNetModels, \
    SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetSenderInterface, SpikeNetReceiverInterface

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



class NetpyneSenderInterface(NetpyneOutputInterface, SpikeNetSenderInterface):

    """NetpyneSenderInterface"""

    pass


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


class NetpyneReceiverInterface(NetpyneInputInterface, SpikeNetReceiverInterface):

    """NetpyneReceiverInterface"""

    pass



class TVBtoNetpyneInterface(NetpyneInputInterface, TVBtoSpikeNetInterface):

    """TVBtoNetpyneInterface class to get data from TVB, transform them,
       and finally set them to NetPyNE, all processes taking place in shared memory.
    """

    pass


class NetpyneToTVBInterface(NetpyneOutputInterface, SpikeNetToTVBInterface):

    """NetpyneToTVBInterface class to get data from NetPyNE, transform them,
       and finally set them to TVB, all processes taking place in shared memory.
    """

    pass


class NetpyneInterfaces(HasTraits):

    """NetpyneInterfaces class holding a list of NetpyneInterface instances"""

    interfaces = List(of=NetpyneInterface)

    @property
    def proxy_gids(self):
        # TODO: implement this one
        raise NotImplementedError

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    @property
    def netpyne_network(self):
        return self.spiking_network

    @property
    def netpyne_instance(self):
        if len(self.interfaces):
            return self.interfaces[0].netpyne_instance
        else:
            return None
        
        
class NetpyneOutputInterfaces(SpikeNetOutputInterfaces, NetpyneInterfaces):

    """NetpyneSenderInterfaces holding a list of NetpyneSenderInterface instances"""

    interfaces = List(of=NetpyneOutputInterface)


class NetpyneInputInterfaces(SpikeNetInputInterfaces, NetpyneInterfaces):

    """NetpyneInputInterfaces holding a list of NetpyneInputInterface instances"""

    interfaces = List(of=NetpyneInputInterface)


class NetpyneSenderInterfaces(SpikeNetSenderInterfaces, NetpyneInterfaces):

    """NetpyneSenderInterfaces holding a list of NetpyneSenderInterface instances"""

    interfaces = List(of=NetpyneSenderInterface)


class NetpyneReceiverInterfaces(SpikeNetReceiverInterfaces, NetpyneInterfaces):

    """NetpyneReceiverInterfaces holding a list of NetpyneReceiverInterface instances"""

    interfaces = List(of=NetpyneReceiverInterface)



class TVBtoNetpyneInterfaces(TVBOutputInterfaces, NetpyneInputInterfaces):

    """TVBtoNetpyneInterfaces class holding a list of TVBtoNetpyneInterface instances"""

    interfaces = List(of=TVBtoNetpyneInterface)


class NetpyneToTVBInterfaces(TVBReceiverInterfaces, NetpyneOutputInterfaces):
    """NetpyneToTVBInterfaces class holding a list of NetpynetoTVBInterface instances"""

    interfaces = List(of=NetpyneToTVBInterface)

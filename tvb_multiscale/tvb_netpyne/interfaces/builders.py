from logging import Logger
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.builders import \
    SpikeNetProxyNodesBuilder, SpikeNetInterfaceBuilder, SpikeNetRemoteInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_netpyne.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_netpyne.interfaces.interfaces import \
    NetpyneOutputInterface, NetpyneInputInterface, \
    NetpyneSenderInterface, NetpyneReceiverInterface, \
    TVBtoNetpyneInterface, NetpyneToTVBInterface
from tvb_multiscale.tvb_netpyne.interfaces.io import \
    NetpyneSpikeRecorderSet, NetpyneSpikeRecorderTotalSet, \
    NetpynePoissonGeneratorSet
from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device


TVBtoNetpyneModels = TVBtoSpikeNetModels
NetpyneToTVBModels = SpikeNetToTVBModels


class NetpyneInputProxyModels(Enum):
    RATE = NetpynePoissonGeneratorSet
    SPIKES = None


class NetpyneOutputProxyModels(Enum):
    SPIKES = NetpyneSpikeRecorderSet
    SPIKES_MEAN = NetpyneSpikeRecorderTotalSet


class DefaultTVBtoNetpyneProxyModels(object):
    RATE = NetpyneInputProxyModels.RATE.name
    SPIKES = None


class DefaultNetpyneToTVBProxyModels(object):
    SPIKES = NetpyneOutputProxyModels.SPIKES_MEAN.name


class NetpyneProxyNodesBuilder(SpikeNetProxyNodesBuilder):

    """NetpyneProxyNodesBuilder class"""

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
    def spikeNet_min_delay(self):
        return self.netpyne_instance.minDelay

    @property
    def netpyne_min_delay(self):
        return self.netpyne_instance.minDelay

    def _build_and_connect_devices(self, interface, **kwargs):
        return build_and_connect_devices(interface, create_device, connect_device,
                                         self.spiking_network.brain_regions,
                                         self.config, netpyne_instance=self.netpyne_instance, **kwargs)

    def _build_and_connect_output_devices(self, interface, **kwargs):
        return self._build_and_connect_devices(interface, **kwargs)

    def _build_and_connect_input_devices(self, interface, **kwargs):
        isTvb = self.is_tvb_coupling_interface(interface)
        scale = self.synaptic_weight_scale_func(is_coupling_mode_tvb=isTvb)
        interface["weights"] = interface["weights"] * scale
        return self._build_and_connect_devices(interface, **kwargs)

    def _default_receptor_type(self, source_node, target_node):
        return None

    @property
    def _default_min_delay(self):
        return self.netpyne_min_delay


class NetpyneInterfaceBuilder(NetpyneProxyNodesBuilder, SpikeNetInterfaceBuilder):
    """NetpyneInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoNetpyneModels
    _spikeNet_to_tvb_models = NetpyneToTVBModels

    _default_tvb_to_spikeNet_proxy_models = DefaultTVBtoNetpyneProxyModels
    _default_spikeNet_to_tvb_proxy_models = DefaultNetpyneToTVBProxyModels

    _input_proxy_models = NetpyneInputProxyModels
    _output_proxy_models = NetpyneOutputProxyModels

    _output_interface_type = NetpyneOutputInterface
    _input_interface_type = NetpyneInputInterface

    _output_interfaces_type = NetpyneOutputInterfaces
    _input_interfaces_type = NetpyneInputInterfaces

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    def _get_tvb_delays(self):
        return np.maximum(self.spiking_dt,
                          SpikeNetInterfaceBuilder._get_tvb_delays(self) - self.spiking_dt).astype("float32")
        
        
class NetpyneRemoteInterfaceBuilder(NetpyneInterfaceBuilder, SpikeNetRemoteInterfaceBuilder):

    """NetpyneRemoteInterfaceBuilder class"""

    _output_interface_type = NetpyneSenderInterface
    _input_interface_type = NetpyneReceiverInterface

    _output_interfaces_type = NetpyneSenderInterfaces
    _input_interfaces_type = NetpyneReceiverInterfaces
    
    def configure(self):
        SpikeNetRemoteInterfaceBuilder.configure(self)
        
        
class TVBNetpyneInterfaceBuilder(NetpyneProxyNodesBuilder, TVBSpikeNetInterfaceBuilder):

    """TVBNetpyneInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoNetpyneModels
    _spikeNet_to_TVB_models = NetpyneToTVBModels

    _default_spikeNet_to_tvb_proxy_models = DefaultNetpyneToTVBProxyModels
    _default_tvb_to_spikeNet_proxy_models = DefaultTVBtoNetpyneProxyModels

    _input_proxy_models = NetpyneOutputProxyModels  # Input to SpikeNet is output of TVB
    _output_proxy_models = NetpyneInputProxyModels  # Output of SpikeNet is input to TVB

    _output_interface_type = TVBtoNetpyneInterface
    _input_interface_type = NetpyneToTVBInterface

    _output_interfaces_type = TVBtoNetpyneInterfaces
    _input_interfaces_type = NetpyneToTVBInterfaces

    def _get_tvb_delays(self):
        return (np.maximum(1,
                           np.rint((TVBSpikeNetInterfaceBuilder._get_tvb_delays(self)
                                    - self.tvb_min_delay + self.spiking_dt)/self.spiking_dt).astype("i")
                           ) * self.spiking_dt).astype("float32")

    def build(self):
        # NetPyNE model is built in two steps.
        # First need to create declarative-style specification for both spiking network itself
        # and TVB-Netpyne proxy devides (interfaces):
        result = super(TVBNetpyneInterfaceBuilder, self).build()
        # once done, network can be instantiated based on the specification:
        self.netpyne_instance.createNetwork()

        self.netpyne_network.tvb_cosimulator = self.tvb_cosimulator

        return result

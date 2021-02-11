# -*- coding: utf-8 -*-
import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr

from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import \
    SpikeNetRemoteInterfaceBuilder, SpikeNetTransformerInterfaceBuilder,  \
    SpikeNetOutputTransformerInterfaceBuilder, SpikeNetInputTransformerInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_nest.interfaces.interfaces import \
    NESTOutputInterfaces, NESTInputInterfaces, \
    NESTSenderInterface, NESTReceiverInterface, \
    NESTTransformerSenderInterface, NESTReceiverTransformerInterface, \
    TVBtoNESTInterfaces, NESTtoTVBInterfaces, \
    TVBtoNESTInterface, NESTtoTVBInterface

from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device


class NESTInterfaceBuilder(HasTraits):

    """NESTInterfaceBuilder class"""

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
    def config(self):
        return self.spiking_network.config

    @property
    def spikeNet_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def nest_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    def _build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, nest_instance=self.nest_instance)

    @property
    def _default_receptor_type(self):
        return 0

    @property
    def _default_min_delay(self):
        return self.nest_min_delay


class NESTRemoteInterfaceBuilder(SpikeNetRemoteInterfaceBuilder, NESTInterfaceBuilder):

    """NESTRemoteInterfaceBuilder class"""

    _spikeNet_output_interfaces_type = NESTOutputInterfaces
    _spikeNet_input_interfaces_type = NESTInputInterfaces

    _spikeNet_output_types = NESTSenderInterface
    _spikeNet_input_types = NESTReceiverInterface

    def configure(self):
        SpikeNetRemoteInterfaceBuilder.configure(self)
        NESTInterfaceBuilder.configure(self)


class NESTTransformerInterfaceBuilder(SpikeNetTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTTransformerInterfaceBuilder class"""

    _spikeNet_output_interfaces_type = NESTOutputInterfaces
    _spikeNet_input_interfaces_type = NESTInputInterfaces

    _spikeNet_output_types = NESTTransformerSenderInterface
    _spikeNet_input_types = NESTReceiverTransformerInterface

    def configure(self):
        SpikeNetTransformerInterfaceBuilder.configure(self)
        NESTInterfaceBuilder.configure(self)


class NESTOutputTransformerInterfaceBuilder(SpikeNetOutputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTOutputTransformerInterfaceBuilder class"""

    _spikeNet_output_interfaces_type = NESTOutputInterfaces
    _spikeNet_input_interfaces_type = NESTInputInterfaces

    _spikeNet_output_types = NESTTransformerSenderInterface
    _spikeNet_input_types = NESTReceiverInterface

    def configure(self):
        SpikeNetOutputTransformerInterfaceBuilder.configure(self)
        NESTInterfaceBuilder.configure(self)


class NESTInputTransformerInterfaceBuilder(SpikeNetInputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTInputTransformerInterfaceBuilder abstract base class"""

    _spikeNet_output_interfaces_type = NESTOutputInterfaces
    _spikeNet_input_interfaces_type = NESTInputInterfaces

    _spikeNet_output_types = NESTSenderInterface
    _spikeNet_input_types = NESTReceiverTransformerInterface

    def configure(self):
        SpikeNetInputTransformerInterfaceBuilder.configure(self)
        NESTInterfaceBuilder.configure(self)


class TVBNESTInterfaceBuilder(TVBSpikeNetInterfaceBuilder, NESTInterfaceBuilder):

    """TVBNESTInterfaceBuilder abstract base class"""

    _tvb_output_interfaces_type = TVBtoNESTInterfaces
    _tvb_input_interfaces_type = NESTtoTVBInterfaces

    _tvb_to_spikeNet_interface_types = TVBtoNESTInterface
    _spikeNet_to_tvb_interface_types = NESTtoTVBInterface

    def configure(self):
        TVBSpikeNetInterfaceBuilder.configure(self)
        NESTInterfaceBuilder.configure(self)

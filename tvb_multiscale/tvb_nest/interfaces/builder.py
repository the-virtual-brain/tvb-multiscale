# -*- coding: utf-8 -*-

from enum import Enum

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
from tvb_multiscale.tvb_nest.interfaces.io import NESTEventsFromSpikeRecorder, NESTSpikeRecorderFile, \
    NESTSpikeGeneratorSetter, NESTInhomogeneousPoissonGeneratorSetter, NESTStepCurrentGeneratorSetter
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.devices import NE
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device


class NESTInputProxyModels(Enum):
    RATE = NESTInhomogeneousPoissonGeneratorSetter
    SPIKES = NESTSpikeGeneratorSetter
    CURRENT = NESTStepCurrentGeneratorSetter


class NESTOutputProxyModels(Enum):
    SPIKES = NESTEventsFromSpikeRecorder
    SPIKES_FILE = NESTSpikeRecorderFile


class NESTInterfaceBuilder(HasTraits):

    """NESTInterfaceBuilder class"""

    _spikeNet_output_proxy_types = [val.value for val in NESTOutputProxyModels.__members__.values()]
    _spikeNet_input_proxy_types = [val.value for val in NESTInputProxyModels.__members__.values()]

    spiking_network = Attr(label="NEST Network",
                           doc="""The instance of NESTNetwork class""",
                           field_type=NESTNetwork,
                           required=True)

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

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
    def _default_receptor_type(self, source_node, target_node):
        return 0

    @property
    def _default_min_delay(self):
        return self.nest_min_delay

    def configure(self):
        for interface in self.input_interfaces:
            model = interface.get("proxy_model", interface.get("model", "SPIKES"))
            if isinstance(model, string_types):
                model = model.upper()
                interface["proxy_model"] = getattr(NESTInputProxyModels, model).value
            else:
                assert model in self._spikeNet_input_proxy_types
        for interface in self.output_interfaces:
            model = interface.get("proxy_model", interface.get("model", "SPIKES")).upper()
            if isinstance(model, string_types):
                model = model.upper()
                interface["proxy_model"] = getattr(NESTOutputProxyModels, model).value
            else:
                assert model in self._spikeNet_output_proxy_types


class NESTRemoteInterfaceBuilder(SpikeNetRemoteInterfaceBuilder, NESTInterfaceBuilder):

    """NESTRemoteInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTSenderInterface
    _input_interface_type = NESTReceiverInterface

    _spikeNet_output_proxy_types = NESTInterfaceBuilder._spikeNet_output_proxy_types
    _spikeNet_input_proxy_types = NESTInterfaceBuilder._spikeNet_input_proxy_types

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetRemoteInterfaceBuilder.configure(self)


class NESTTransformerInterfaceBuilder(SpikeNetTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTTransformerSenderInterface
    _input_interface_type = NESTReceiverTransformerInterface

    _spikeNet_output_proxy_types = NESTInterfaceBuilder._spikeNet_output_proxy_types
    _spikeNet_input_proxy_types = NESTInterfaceBuilder._spikeNet_input_proxy_types

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetTransformerInterfaceBuilder.configure(self)


class NESTOutputTransformerInterfaceBuilder(SpikeNetOutputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTOutputTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTTransformerSenderInterface
    _input_interface_type = NESTReceiverInterface

    _spikeNet_output_proxy_types = NESTInterfaceBuilder._spikeNet_output_proxy_types
    _spikeNet_input_proxy_types = NESTInterfaceBuilder._spikeNet_input_proxy_types

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetOutputTransformerInterfaceBuilder.configure(self)


class NESTInputTransformerInterfaceBuilder(SpikeNetInputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTInputTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTSenderInterface
    _input_interface_type = NESTReceiverTransformerInterface

    _spikeNet_output_proxy_types = NESTInterfaceBuilder._spikeNet_output_proxy_types
    _spikeNet_input_proxy_types = NESTInterfaceBuilder._spikeNet_input_proxy_types

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetInputTransformerInterfaceBuilder.configure(self)


class TVBNESTInterfaceBuilder(TVBSpikeNetInterfaceBuilder, NESTInterfaceBuilder):

    """TVBNESTInterfaceBuilder class"""

    _output_interfaces_type = TVBtoNESTInterfaces
    _input_interfaces_type = NESTtoTVBInterfaces

    _output_interface_type = TVBtoNESTInterface
    _input_interface_type = NESTtoTVBInterface

    _spikeNet_output_proxy_types = NESTInterfaceBuilder._spikeNet_output_proxy_types
    _spikeNet_input_proxy_types = NESTInterfaceBuilder._spikeNet_input_proxy_types

    _output_interfaces_type = TVBtoNESTInterfaces
    _input_interfaces_type = NESTtoTVBInterfaces

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        TVBSpikeNetInterfaceBuilder.configure(self)

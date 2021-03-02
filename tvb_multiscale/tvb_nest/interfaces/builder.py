# -*- coding: utf-8 -*-

from six import string_types
from enum import Enum

from tvb.basic.neotraits.api import HasTraits, Attr, List

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
from tvb_multiscale.tvb_nest.interfaces.io import NESTSpikeRecorderSet, \
    NESTSpikeGeneratorSet, NESTInhomogeneousPoissonGeneratorSet, NESTStepCurrentGeneratorSet, \
    NESTParrotSpikeGeneratorSet, NESTParrotInhomogeneousPoissonGeneratorSet
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device


class NESTInputProxyModels(Enum):
    RATE = NESTInhomogeneousPoissonGeneratorSet
    PARROT_RATE = NESTParrotInhomogeneousPoissonGeneratorSet
    RATE_TO_SPIKES = NESTSpikeGeneratorSet
    PARROT_RATE_TO_SPIKES = NESTParrotSpikeGeneratorSet
    CURRENT = NESTStepCurrentGeneratorSet


class NESTDefaultInputProxyModels(Enum):
    RATE = NESTParrotInhomogeneousPoissonGeneratorSet
    RATE_TO_SPIKES = NESTParrotSpikeGeneratorSet
    CURRENT = NESTStepCurrentGeneratorSet


class NESTOutputProxyModels(Enum):
    SPIKES_TO_RATE = NESTSpikeRecorderSet


class NESTDefaultOutputProxyModels(Enum):
    SPIKES_TO_RATE = NESTSpikeRecorderSet


NEST_OUTPUT_PROXY_TYPES = tuple([val.value for val in NESTOutputProxyModels.__members__.values()])
NEST_OUTPUT_PROXY_MODELS = tuple([val.name for val in NESTDefaultOutputProxyModels.__members__.values()])

NEST_INPUT_PROXY_TYPES = tuple([val.value for val in NESTInputProxyModels.__members__.values()])
NEST_INPUT_PROXY_MODELS = tuple([val.name for val in NESTDefaultInputProxyModels.__members__.values()])


class NESTInterfaceBuilder(HasTraits):

    """NESTInterfaceBuilder class"""

    spiking_network = Attr(label="NEST Network",
                           doc="""The instance of NESTNetwork class""",
                           field_type=NESTNetwork,
                           required=True)

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

    _default_input_proxy_model = NESTDefaultInputProxyModels.RATE.name
    _default_output_proxy_model = NESTDefaultOutputProxyModels.SPIKES_TO_RATE.name

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

    def _default_receptor_type(self, source_node, target_node):
        return 0

    @property
    def _default_min_delay(self):
        return self.nest_min_delay

    def configure(self):
        for interface in self.input_interfaces:
            model = interface.get("proxy_model", interface.get("model", self._default_input_proxy_model))
            if isinstance(model, string_types):
                model = model.upper()
                assert model in NEST_INPUT_PROXY_MODELS
                interface["proxy_model"] = getattr(NESTDefaultInputProxyModels, model).value
            else:
                assert model in NEST_INPUT_PROXY_TYPES
        for interface in self.output_interfaces:
            model = interface.get("proxy_model", interface.get("model", self._default_output_proxy_model))
            if isinstance(model, string_types):
                model = model.upper()
                assert model in NEST_OUTPUT_PROXY_MODELS
                interface["proxy_model"] = getattr(NESTDefaultOutputProxyModels, model).value
            else:
                assert model in NEST_OUTPUT_PROXY_TYPES
        super(NESTInterfaceBuilder, self).configure()


class NESTRemoteInterfaceBuilder(SpikeNetRemoteInterfaceBuilder, NESTInterfaceBuilder):

    """NESTRemoteInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTSenderInterface
    _input_interface_type = NESTReceiverInterface

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetRemoteInterfaceBuilder.configure(self)


class NESTTransformerInterfaceBuilder(SpikeNetTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTTransformerSenderInterface
    _input_interface_type = NESTReceiverTransformerInterface

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetTransformerInterfaceBuilder.configure(self)


class NESTOutputTransformerInterfaceBuilder(SpikeNetOutputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTOutputTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTTransformerSenderInterface
    _input_interface_type = NESTReceiverInterface

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetOutputTransformerInterfaceBuilder.configure(self)


class NESTInputTransformerInterfaceBuilder(SpikeNetInputTransformerInterfaceBuilder, NESTInterfaceBuilder):

    """NESTInputTransformerInterfaceBuilder class"""

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

    _output_interface_type = NESTSenderInterface
    _input_interface_type = NESTReceiverTransformerInterface

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        SpikeNetInputTransformerInterfaceBuilder.configure(self)


class TVBNESTInterfaceBuilder(TVBSpikeNetInterfaceBuilder, NESTInterfaceBuilder):

    """TVBNESTInterfaceBuilder class"""

    _output_interfaces_type = TVBtoNESTInterfaces
    _input_interfaces_type = NESTtoTVBInterfaces

    _output_interface_type = TVBtoNESTInterface
    _input_interface_type = NESTtoTVBInterface

    def configure(self):
        NESTInterfaceBuilder.configure(self)
        TVBSpikeNetInterfaceBuilder.configure(self)

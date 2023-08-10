# -*- coding: utf-8 -*-
from logging import Logger
from enum import Enum
import pickle

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.base.builders import RemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetProxyNodesBuilder, SpikeNetInterfaceBuilder, \
    SpikeNetTransformerInterfaceBuilder,  SpikeNetRemoteInterfaceBuilder, \
    SpikeNetRemoteTransformerInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.interfaces.interfaces import \
    NESTOutputInterface, NESTInputInterface, \
    NESTOutputTransformerInterface, NESTInputTransformerInterface, \
    NESTSenderInterface, NESTReceiverInterface, \
    NESTTransformerSenderInterface, NESTReceiverTransformerInterface, \
    TVBtoNESTInterface, NESTtoTVBInterface, \
    NESTOutputInterfaces, NESTInputInterfaces, \
    NESTOutputTransformerInterfaces, NESTInputTransformerInterfaces, \
    NESTSenderInterfaces, NESTReceiverInterfaces, \
    NESTTransformerSenderInterfaces, NESTReceiverTransformerInterfaces, \
    TVBtoNESTInterfaces, NESTtoTVBInterfaces
from tvb_multiscale.tvb_nest.interfaces.io import \
    NESTSpikeRecorderSet, NESTSpikeRecorderTotalSet, \
    NESTSpikeGeneratorSet, NESTInhomogeneousPoissonGeneratorSet, NESTStepCurrentGeneratorSet, \
    NESTParrotSpikeGeneratorSet, NESTParrotInhomogeneousPoissonGeneratorSet, \
    NESTVoltmeterSet, NESTVoltmeterMeanSet, NESTVoltmeterTotalSet
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device


TVBtoNESTModels = TVBtoSpikeNetModels
NESTtoTVBModels = SpikeNetToTVBModels


class NESTInputProxyModels(Enum):
    RATE = NESTInhomogeneousPoissonGeneratorSet
    RATE_TO_SPIKES = NESTParrotInhomogeneousPoissonGeneratorSet
    SPIKES = NESTSpikeGeneratorSet
    PARROT_SPIKES = NESTParrotSpikeGeneratorSet
    CURRENT = NESTStepCurrentGeneratorSet


class NESTOutputProxyModels(Enum):
    SPIKES = NESTSpikeRecorderSet
    SPIKES_MEAN = NESTSpikeRecorderTotalSet
    POTENTIAL = NESTVoltmeterSet
    POTENTIAL_MEAN = NESTVoltmeterMeanSet
    POTENTIAL_TOTAL = NESTVoltmeterTotalSet


class DefaultTVBtoNESTModels(Enum):
    RATE = NESTInputProxyModels.RATE.name
    SPIKES = NESTInputProxyModels.SPIKES.name
    CURRENT = NESTInputProxyModels.CURRENT.name


class DefaultNESTtoTVBModels(Enum):
    SPIKES = NESTOutputProxyModels.SPIKES_MEAN.name
    POTENTIAL = NESTOutputProxyModels.POTENTIAL_MEAN.name


class NESTProxyNodesBuilder(SpikeNetProxyNodesBuilder):

    """NESTProxyNodesBuilder class"""

    spiking_network = Attr(label="NEST Network",
                           doc="""The instance of NESTNetwork class""",
                           field_type=NESTNetwork,
                           required=False)

    @property
    def nest_network(self):
        return self.spiking_network

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    @property
    def spikeNet_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def nest_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    def _build_and_connect_devices(self, interface, **kwargs):
        return build_and_connect_devices(interface, create_device, connect_device,
                                         self.spiking_network.brain_regions,
                                         self.config, nest_instance=self.nest_instance, **kwargs)

    def _build_and_connect_output_devices(self, interface, **kwargs):
        if "meter" in interface["model"]:  # TODO: Find a better way to do this!
            interface["params"]["interval"] = interface["params"].get("interval", self.tvb_dt)
        return self._build_and_connect_devices(interface, **kwargs)

    def _build_and_connect_input_devices(self, interface, **kwargs):
        return self._build_and_connect_devices(interface, **kwargs)

    def _default_receptor_type(self, source_node, target_node):
        return 0

    @property
    def _default_min_delay(self):
        return self.nest_min_delay


class NESTInterfaceBuilder(NESTProxyNodesBuilder, SpikeNetInterfaceBuilder):

    """NESTInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoNESTModels
    _spikeNet_to_tvb_models = NESTtoTVBModels

    _default_tvb_to_spikeNet_models = DefaultTVBtoNESTModels
    _default_spikeNet_to_tvb_models = DefaultNESTtoTVBModels

    _input_proxy_models = NESTInputProxyModels
    _output_proxy_models = NESTOutputProxyModels

    _output_interface_type = NESTOutputInterface
    _input_interface_type = NESTInputInterface

    _output_interfaces_type = NESTOutputInterfaces
    _input_interfaces_type = NESTInputInterfaces

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


class NESTRemoteInterfaceBuilder(NESTInterfaceBuilder, SpikeNetRemoteInterfaceBuilder):

    """NESTRemoteInterfaceBuilder class"""

    _output_interface_type = NESTSenderInterface
    _input_interface_type = NESTReceiverInterface

    _output_interfaces_type = NESTSenderInterfaces
    _input_interfaces_type = NESTReceiverInterfaces

    def configure(self):
        SpikeNetRemoteInterfaceBuilder.configure(self)


class NESTMPIInterfaceBuilder(NESTRemoteInterfaceBuilder):

    # This builder doesn't need to create any Communicators,
    # since communication in and out of NEST is handled by NEST MPI automatically.
    # But, it has to write to a file the gids of each interface's NEST proxy devices created

    _default_remote_sender_type = None
    _default_remote_receiver_type = None

    def configure(self):
        SpikeNetInterfaceBuilder.configure(self)
        super(RemoteInterfaceBuilder, self).configure()

    def _get_output_interface_arguments(self, interface, ii=0):
        proxy_params = {"record_to": "MPI",
                        "label": self._file_path(self.output_label) + "%d" % ii}
        if "proxy_params" in interface:
            interface["proxy_params"].update(proxy_params)
        else:
            interface["proxy_params"] = proxy_params
        interface = SpikeNetInterfaceBuilder._get_output_interface_arguments(self, interface, ii)
        if "proxy_inds" in interface:
            del interface["proxy_inds"]
        return interface

    def _get_input_interface_arguments(self, interface, ii=0):
        interface["proxy_params"].update({"stimulus_source": "MPI",
                                          "label": self._file_path(self.input_label) + "%d" % ii})
        interface = SpikeNetInterfaceBuilder._get_input_interface_arguments(self, interface, ii)
        # We don't need these anymore for interfaces without TVB:
        if "proxy_inds" in interface:
            del interface["proxy_inds"]
        if 'coupling_mode' in interface:
            del interface['coupling_mode']
        return interface

    def write_proxies_gids(self):
        out_gids = []
        for ii, interface in enumerate(self.output_interfaces):
            out_gids.append(interface.proxy_gids)
        in_gids = []
        for ii, interface in enumerate(self.input_interfaces):
            in_gids.append(interface.proxy_gids)
        with open(self._file_path(self.output_label) + "_gids.pkl", 'wb') as file:
            pickle.dump(out_gids, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self._file_path(self.input_label) + "_gids.pkl", 'wb') as file:
            pickle.dump(in_gids, file, protocol=pickle.HIGHEST_PROTOCOL)

    def build(self):
        output_interfaces, input_interfaces = super(NESTMPIInterfaceBuilder, self).build()
        self.write_proxies_gids()
        return output_interfaces, input_interfaces


class NESTTransformerInterfaceBuilder(NESTInterfaceBuilder, SpikeNetTransformerInterfaceBuilder):

    """NESTTransformerInterfaceBuilder class"""

    _output_interface_type = NESTOutputTransformerInterface
    _input_interface_type = NESTInputTransformerInterface

    _output_interfaces_type = NESTOutputTransformerInterfaces
    _input_interfaces_type = NESTInputTransformerInterfaces

    def configure(self):
        SpikeNetTransformerInterfaceBuilder.configure(self)


class NESTRemoteTransformerInterfaceBuilder(NESTInterfaceBuilder, SpikeNetRemoteTransformerInterfaceBuilder):

    """NESTTransformerInterfaceBuilder class"""

    _output_interface_type = NESTTransformerSenderInterface
    _input_interface_type = NESTReceiverTransformerInterface

    _output_interfaces_type = NESTTransformerSenderInterfaces
    _input_interfaces_type = NESTReceiverTransformerInterfaces

    def configure(self):
        SpikeNetRemoteTransformerInterfaceBuilder.configure(self)


class TVBNESTInterfaceBuilder(NESTProxyNodesBuilder, TVBSpikeNetInterfaceBuilder):

    """TVBNESTInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoNESTModels
    _spikeNet_to_TVB_models = NESTtoTVBModels

    _default_spikeNet_to_tvb_models = DefaultNESTtoTVBModels
    _default_tvb_to_spikeNet_models = DefaultTVBtoNESTModels

    _input_proxy_models = NESTOutputProxyModels  # Input to SpikeNet is output of TVB
    _output_proxy_models = NESTInputProxyModels  # Output of SpikeNet is input to TVB

    _output_interface_type = TVBtoNESTInterface
    _input_interface_type = NESTtoTVBInterface

    _output_interfaces_type = TVBtoNESTInterfaces
    _input_interfaces_type = NESTtoTVBInterfaces

    def configure(self):
        TVBSpikeNetInterfaceBuilder.configure(self)

    def _get_tvb_delays(self):
        return (np.maximum(1,
                           np.rint((TVBSpikeNetInterfaceBuilder._get_tvb_delays(self)
                                    - self.synchronization_time + self.spiking_dt)/self.spiking_dt).astype("i")
                           ) * self.spiking_dt).astype("float32")

# -*- coding: utf-8 -*-

from logging import Logger
from abc import ABCMeta, ABC

import numpy as np

from tvb.basic.neotraits._attr import Attr
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.base.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import \
    SpikeNetProxyNodesBuilder, DefaultTVBtoSpikeNetModels, DefaultSpikeNetToTVBModels
from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers
from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import \
    TVBOutputInterfaces, TVBInputInterfaces, TVBOutputInterface, TVBInputInterface, \
    TVBSenderInterface, TVBReceiverInterface, TVBTransformerSenderInterface, TVBReceiverTransformerInterface, \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBtoSpikeNetInterfaces, SpikeNetToTVBInterfaces, \
    TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator


class TVBInterfaceBuilder(InterfaceBuilder):

    """TVBInterfaceBuilder abstract base class"""

    _output_interface_type = TVBOutputInterface
    _input_interface_type = TVBInputInterface

    _output_interfaces_type = TVBOutputInterfaces
    _input_interfaces_type = TVBInputInterfaces

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

    tvb_cosimulator = Attr(label="TVB simulator",
                           doc="""The instance of TVB simulator""",
                           field_type=CoSimulator,
                           required=True)

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                  if the co-simulator nodes are modelled exclusively by the co-simulator, 
                                  i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    _default_out_proxy_inds = np.array([])

    @property
    def tvb_dt(self):
        if self.tvb_cosimulator is None:
            return self.config.DEFAULT_DT
        return self.tvb_cosimulator.integrator.dt

    @property
    def tvb_model(self):
        if self.tvb_cosimulator is None:
            return ""
        return self.tvb_cosimulator.model

    @property
    def tvb_model_state_variables(self):
        if self.tvb_cosimulator is None:
            return []
        return self.tvb_cosimulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        if self.tvb_cosimulator is None:
            return []
        return self.tvb_cosimulator.model.cvar

    @property
    def number_of_regions(self):
        if self.tvb_cosimulator is None:
            return 0
        return self.tvb_cosimulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        if self.tvb_cosimulator is None:
            return np.array([])
        return self.tvb_cosimulator.connectivity.region_labels

    @property
    def tvb_coupling_a(self):
        if self.tvb_cosimulator is None:
            return np.array([1.0/256])
        return self.tvb_cosimulator.coupling.a

    @property
    def tvb_weights(self):
        if self.tvb_cosimulator is None:
            return np.zeros((0, 0))
        return self.tvb_cosimulator.connectivity.weights

    @property
    def tvb_delays(self):
        if self.tvb_cosimulator is None:
            return self.tvb_dt * np.ones((0, 0))
        return self.tvb_cosimulator.connectivity.delays

    def _proxy_inds(self, interfaces):
        return np.unique(self._only_inds_for_interfaces(interfaces, "proxy_inds", self.region_labels))

    @property
    def out_proxy_inds(self):
        return self._proxy_inds(self.output_interfaces)

    @property
    def in_proxy_inds(self):
        return self._proxy_inds(self.input_interfaces)

    def _voi_inds_labels_for_interface(self, interface):
        voi_inds = np.array(self._only_inds(interface["voi"], self.tvb_model_state_variables))
        voi_labels = np.array(self.tvb_model_state_variables)[voi_inds]
        return voi_inds, voi_labels

    def _voi_inds(self, interfaces):
        return np.unique(self._only_inds_for_interfaces(interfaces, "voi", self.tvb_model_state_variables))

    @property
    def out_voi_inds(self):
        return self._voi_inds(self.output_interfaces)

    @property
    def in_voi_inds(self):
        return self._voi_inds(self.input_interfaces)

    @property
    def number_of_out_voi(self):
        return len(self.out_voi_inds)

    @property
    def number_of_in_voi(self):
        return len(self.in_voi_inds)

    def region_label_to_ind(self, labels):
        return self._label_to_ind(labels, self.region_labels)

    @property
    def out_proxy_labels(self):
        return self.region_labels[self.out_proxy_inds]

    @property
    def in_proxy_labels(self):
        return self.region_labels[self.in_proxy_inds]

    def voi_label_to_ind(self, voi):
        return self._label_to_ind(voi, self.tvb_cosimulator.model.state_variables)

    @property
    def out_voi_labels(self):
        return self.tvb_model_state_variables[self.out_voi_inds]

    @property
    def in_voi_labels(self):
        return self.tvb_model_state_variables[self.in_voi_inds]

    def _configure_proxys_vois(self, interfaces, default_proxy_inds):
        for interface in interfaces:
            interface["proxy_inds"] = np.array(
                self._only_inds(
                    ensure_list(interface.pop("proxy_inds", interface.pop("proxy", default_proxy_inds))),
                    self.region_labels))
            assert (interface["proxy_inds"] >= 0).all and (interface["proxy_inds"] <= self.number_of_regions).all
            voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
            interface["voi"] = voi_inds
            interface["voi_labels"] = voi_labels

    def configure(self):
        super(TVBInterfaceBuilder, self).configure()
        self._default_out_proxy_inds = np.arange(self.number_of_regions).astype('i').tolist()
        self._configure_proxys_vois(self.input_interfaces, self.proxy_inds)
        if self.exclusive_nodes:
            for proxy_ind in self.in_proxy_inds:
                self._default_out_proxy_inds.remove(proxy_ind)
        self._default_out_proxy_inds = np.array(self._default_out_proxy_inds)
        self._configure_proxys_vois(self.output_interfaces, self._default_out_proxy_inds)
    
    def _get_output_interface_arguments(self, interface):
        interface.update({"monitor_ind": interface.get("monitor_ind", 0)})
        return interface

    def _get_input_interface_arguments(self, interface):
        return interface

    def build(self):
        self.build_interfaces()
        self.tvb_cosimulator.exclusive = self.exclusive_nodes
        self.tvb_cosimulator.output_interfaces = self._output_interfaces_type(interfaces=self._output_interfaces)
        self.tvb_cosimulator.input_interfaces = self._input_interfaces_type(interfaces=self._input_interfaces)
        return self.tvb_cosimulator


class TVBRemoteInterfaceBuilder(TVBInterfaceBuilder):

    """TVBRemoteInterfaceBuilder class"""

    _output_interface_type = TVBSenderInterface
    _input_interface_type = TVBReceiverInterface
    
    _remote_sender_types = [val.value for val in RemoteSenders.__members__.values()]
    _remote_receiver_types = [val.value for val in RemoteReceivers.__members__.values()]

    def configure(self):
        super(TVBRemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._remote_sender_types, "sender")
        self._assert_input_interfaces_component_config(self._remote_receiver_types, "receiver")

    def _get_output_interface_arguments(self, interface):
        interface = super(TVBRemoteInterfaceBuilder, self)._get_output_interface_arguments(interface)
        interface["communicator"] = interface.pop("sender")
        return interface

    def _get_input_interface_arguments(self, interface):
        interface = super(TVBRemoteInterfaceBuilder, self)._get_input_interface_arguments(interface)
        interface["communicator"] = interface.pop("receiver")
        return interface


class TVBTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder,
                                    TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder):

    """TVBTransfomerInterfaceBuilder class"""

    _output_interface_type = TVBTransformerSenderInterface
    _input_interface_type = TVBReceiverTransformerInterface

    def configure(self):
        super(TVBTransfomerInterfaceBuilder, self).configure()
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformer(self, self.output_interfaces)
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformer(self, self.input_interfaces)

    def _get_output_interface_arguments(self, interface):
        interface = super(TVBTransfomerInterfaceBuilder, self)._get_output_interface_arguments(interface)
        return interface

    def _get_input_interface_arguments(self, interface):
        interface = TVBTransfomerInterfaceBuilder._get_input_interface_arguments(self, interface)
        return interface


class TVBOutputTransformerInterfaceBuilder(TVBRemoteInterfaceBuilder, TVBtoSpikeNetTransformerBuilder):

    """TVBOutputTransformerInterfaceBuilder class"""

    _output_interface_type = TVBTransformerSenderInterface
    _input_interface_type = TVBReceiverInterface

    def configure(self):
        self.configure_and_build_transformer(self)
        super(TVBOutputTransformerInterfaceBuilder, self).configure()

    def _get_output_interface_arguments(self, interface):
        interface.update(super(TVBOutputTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface))
        return interface


class TVBInputTransformerInterfaceBuilder(TVBRemoteInterfaceBuilder, SpikeNetToTVBTransformerBuilder):

    """TVBInputTransformerInterfaceBuilder class"""

    _output_interface_type = TVBSenderInterface
    _input_interface_type = TVBReceiverTransformerInterface

    def configure(self):
        super(TVBInputTransformerInterfaceBuilder, self).configure()
        self.configure_and_build_transformer(self)

    def _get_input_interface_arguments(self, interface):
        interface = super(TVBInputTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface)
        return interface


class TVBSpikeNetInterfaceBuilder(TVBInterfaceBuilder, SpikeNetProxyNodesBuilder,
                                  TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder, ABC):
    __metaclass__ = ABCMeta

    """TVBSpikeNetInterfaceBuilder abstract base class"""

    _tvb_to_spikeNet_models = TVBtoSpikeNetModels
    _spikeNet_to_tvb_models = SpikeNetToTVBModels

    _default_nest_to_tvb_models = DefaultSpikeNetToTVBModels
    _default_tvb_to_nest_models = DefaultTVBtoSpikeNetModels

    _input_proxy_models = None   # Input to SpikeNet is output of TVB
    _output_proxy_models = None  # Output of SpikeNet is input to TVB

    _output_interfaces_type = TVBtoSpikeNetInterfaces
    _input_interfaces_type = SpikeNetToTVBInterfaces

    _output_interface_type = TVBtoSpikeNetInterface
    _input_interface_type = SpikeNetToTVBInterface

    @property
    def tvb_nodes_inds(self):
        return self._default_out_proxy_inds

    @property
    def spiking_nodes_inds(self):
        return self.in_proxy_inds

    def configure(self):
        TVBInterfaceBuilder.configure(self)
        SpikeNetProxyNodesBuilder.configure(self)
        self._configure_proxy_models(self.output_interfaces, self._tvb_to_spikeNet_models,
                                     self._default_tvb_to_nest_models, self._output_proxy_models)
        self._configure_proxy_models(self.input_interfaces, self._spikeNet_to_tvb_models,
                                     self._default_nest_to_tvb_models, self._input_proxy_models)
        # From TVBInterfaceBuilder to
        # SpikeNetProxyNodesBuilder, TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder:
        self.dt = self.tvb_dt
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformer(self, self.output_interfaces)
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformer(self, self.input_interfaces)

    def _get_spikeNet_interface_arguments(self, interface):
        interface.update({"spiking_network": self.spiking_network, "populations": np.array(interface["populations"])})

    def _get_spikeNet_output_interface_arguments(self, interface):
        self._get_spikeNet_interface_arguments(interface)
        interface["dt"] = self.tvb_dt
        self._get_spiking_proxy_inds_for_input_interface(interface, self.exclusive_nodes)
        self._build_spikeNet_to_tvb_interface_proxy_nodes(interface)

    def _get_spikeNet_input_interface_arguments(self, interface):
        self._get_spikeNet_interface_arguments(interface)
        self._get_spiking_proxy_inds_for_output_interface(interface, self.exclusive_nodes)
        self._build_tvb_to_spikeNet_interface_proxy_nodes(interface)

    def _get_output_interface_arguments(self, interface):
        self._get_spikeNet_input_interface_arguments(
            TVBInterfaceBuilder._get_output_interface_arguments(self, interface))
        return interface

    def _get_input_interface_arguments(self, interface):
        self._get_spikeNet_output_interface_arguments(
            TVBInterfaceBuilder._get_input_interface_arguments(self, interface))
        return interface

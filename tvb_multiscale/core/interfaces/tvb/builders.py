# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.interfaces.base.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetProxyNodesBuilder
from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers
from tvb_multiscale.core.interfaces.tvb.transformers import TVBTransformers
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBOutputInterfaces, TVBInputInterfaces, \
    TVBSenderInterface, TVBReceiverInterface, TVBTransformerSenderInterface, TVBReceiverTransformerInterface, \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBtoSpikeNetInterfaces, SpikeNetToTVBInterfaces
from tvb_multiscale.core.tvb.cosimulator import CoSimulator


LOG = initialize_logger(__name__)


class TVBInterfaceBuilder(InterfaceBuilder):
    __metaclass__ = ABCMeta

    """TVBInterfaceBuilder abstract base class"""

    _tvb_output_interfaces_type = TVBOutputInterfaces
    _tvb_input_interfaces_type = TVBInputInterfaces

    tvb_simulator = Attr(label="TVB simulator",
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

    _default_out_proxy_inds = []

    @property
    def tvb_dt(self):
        return self.tvb_simulator.integrator.dt

    @property
    def tvb_model(self):
        return self.tvb_simulator.model

    @property
    def tvb_model_state_variables(self):
        return self.tvb_simulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        return self.tvb_simulator.model.cvar

    @property
    def number_of_regions(self):
        return self.tvb_simulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        return self.tvb_simulator.connectivity.region_labels

    @property
    def tvb_coupling_a(self):
        return self.tvb_simulator.coupling.a

    @property
    def tvb_weights(self):
        return self.tvb_simulator.connectivity.weights

    @property
    def tvb_delays(self):
        return self.tvb_simulator.connectivity.delays

    def _proxy_inds(self, interfaces):
        return np.unique(self._only_inds_for_interfaces(interfaces, "proxy", self.region_labels))

    @property
    def out_proxy_inds(self):
        return self._proxy_inds(self.output_interfaces)

    @property
    def in_proxy_inds(self):
        return self._proxy_inds(self.input_interfaces)

    def _voi_inds_labels_for_interface(self, interface):
        voi_inds = np.array(self._only_inds(interface.voi, self.tvb_model_state_variables))
        voi_labels = self.tvb_model_state_variables[voi_inds]
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
        return self._label_to_ind(voi, self.tvb_simulator.model.state_variables)

    @property
    def out_voi_labels(self):
        return self.tvb_model_state_variables[self.out_voi_inds]

    @property
    def in_voi_labels(self):
        return self.tvb_model_state_variables[self.in_voi_inds]

    def configure(self):
        super(TVBInterfaceBuilder, self).configure()
        self._default_out_proxy_inds = np.arange(self.number_of_regions).astype('i').tolist()
        if self.exclusive_nodes:
            for proxy_ind in self.in_proxy_inds:
                self._default_out_proxy_inds.remove(proxy_ind)
        self._default_out_proxy_inds = np.array(self._default_out_proxy_inds)
        if len(self.output_interfaces):
            assert self.out_voi_labels in self.tvb_model_state_variables
            assert self.out_proxy_labels in self.region_labels
        if len(self.input_interfaces):
            assert self.in_voi_labels in self.tvb_model_state_variables
            assert self.in_proxy_labels in self.region_labels

    @abstractmethod
    def build_output_interface(self, interface):
        pass

    @abstractmethod
    def build_input_interface(self, interface):
        pass

    def build_interfaces(self):
        self._output_interfaces = []
        for interface in self.output_interfaces:
            self._output_interfaces.append(self.build_output_interface(interface))
        self._input_interfaces = []
        for interface in self.input_interfaces:
            self._input_interfaces.append(self.build_input_interface(interface))

    def build(self):
        self.tvb_simulator.exclusive = self.exclusive_nodes
        self.build_interfaces()
        self.tvb_simulator.output_interfaces = self._tvb_output_interfaces_type(interfaces=self._output_interfaces)
        self.tvb_simulator.input_interfaces = self._tvb_input_interfaces_type(interfaces=self._input_interfaces)
        return self.tvb_simulator


class TVBRemoteInterfaceBuilder(TVBInterfaceBuilder):

    """TVBRemoteInterfaceBuilder class"""

    _remote_senders_types = [val.value for val in RemoteSenders.__members__.values()]
    _remote_receivers_types = [val.value for val in RemoteReceivers.__members__.values()]

    def configure(self):
        super(TVBRemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._remote_senders_types, "sender")
        self._assert_input_interfaces_component_config(self._remote_receivers_types, "receiver")
        self._assert_input_interfaces_component_config(self._remote_receivers_types, "receiver")

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBSenderInterface(proxy_inds=self._only_inds(interface.get("proxy_inds",
                                                                           self._default_out_proxy_inds),
                                                             self.region_labels),
                                  voi=voi_inds, voi_labels=voi_labels,
                                  communicator=interface["sender"],
                                  monitor_ind=interface.get("monitor_ind", 0))

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBReceiverInterface(proxy_inds=self._only_inds(interface.get("spiking_proxy_inds", self.proxy_inds),
                                                            self.region_labels),
                                    voi=voi_inds, voi_labels=voi_labels,
                                    communicator=interface["receiver"])


class TVBTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    """TVBTransfomerInterfaceBuilder class"""

    _tvb_transformers_types = [val.value for val in TVBTransformers.__members__.values()]

    def configure(self):
        super(TVBTransfomerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "transformer")
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "transformer")

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBTransformerSenderInterface(
                    proxy_inds=self._only_inds(interface.get("proxy_inds", self._default_out_proxy_inds),
                                               self.region_labels),
                    voi=voi_inds, voi_labels=voi_labels,
                    communicator=interface["sender"],
                    monitor_ind=interface.get("monitor_ind", 0),
                    transformer=interface["transformer"])

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBReceiverTransformerInterface(
                    proxy_inds=self._only_inds(interface.get("proxy_inds", self.proxy_inds), self.region_labels),
                    voi=voi_inds, voi_labels=voi_labels,
                    communicator=interface["receiver"],
                    transformer=interface["transformer"])


class TVBOutputTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    """TVBOutputTransfomerInterfaceBuilder class"""

    _tvb_transformers_types = [val.value for val in TVBTransformers.__members__.values()]

    def configure(self):
        super(TVBOutputTransfomerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "transformer")

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBTransformerSenderInterface(
                    proxy_inds=self._only_inds(interface.get("proxy_inds", self._default_out_proxy_inds),
                                               self.region_labels),
                    voi=voi_inds, voi_labels=voi_labels,
                    communicator=interface["sender"],
                    monitor_ind=interface.get("monitor_ind", 0),
                    transformer=interface["transformer"])

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBReceiverInterface(proxy_inds=self._only_inds(interface.get("proxy_inds", self.proxy_inds),
                                                                self.region_labels),
                                    voi=voi_inds, voi_labels=voi_labels,
                                    communicator=interface["receiver"])


class TVBInputTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    """TVBInputTransfomerInterfaceBuilder class"""

    _tvb_transformers_types = [val.value for val in TVBTransformers.__members__.values()]

    def configure(self):
        super(TVBInputTransfomerInterfaceBuilder, self).configure()
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "transformer")

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBSenderInterface(proxy_inds=self._only_inds(interface.get("proxy_inds", self._default_out_proxy_inds),
                                                              self.region_labels),
                                  voi=voi_inds, voi_labels=voi_labels,
                                  communicator=interface["sender"],
                                  monitor_ind=interface.get("monitor_ind", 0))

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        return TVBReceiverTransformerInterface(
                    proxy_inds=self._only_inds(interface.get("proxy_inds", self.proxy_inds), self.region_labels),
                    voi=voi_inds, voi_labels=voi_labels,
                    communicator=interface["receiver"],
                    transformer=interface["transformer"])


class TVBSpikeNetInterfaceBuilder(TVBInterfaceBuilder, SpikeNetProxyNodesBuilder):
    __metaclass__ = ABCMeta

    """TVBSpikeNetInterfaceBuilder abstract base class"""

    _tvb_output_interfaces_type = TVBtoSpikeNetInterfaces
    _tvb_input_interfaces_type = SpikeNetToTVBInterfaces

    _tvb_transformers_types = [val.value for val in TVBTransformers.__members__.values()]
    _tvb_to_spikeNet_interface_types = TVBtoSpikeNetInterface
    _spikeNet_to_tvb_interface_types = SpikeNetToTVBInterface

    @property
    def tvb_nodes_inds(self):
        return self._default_out_proxy_inds

    @property
    def spiking_nodes_inds(self):
        return self.in_proxy_inds

    @property
    @abstractmethod
    def _default_receptor_type(self):
        pass

    @property
    @abstractmethod
    def _default_min_delay(self):
        pass

    def configure(self):
        SpikeNetProxyNodesBuilder.configure(self)
        TVBInterfaceBuilder.configure(self)
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "transformer")
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "transformer")
        self._assert_output_interfaces_component_config(self._tvb_to_spikeNet_interface_types, "model")
        self._assert_input_interfaces_component_config(self._spikeNet_to_tvb_interface_types, "model")

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        tvb_nodes_inds, spiking_proxy_inds = \
            self._get_tvb_nodes_spiking_proxy_inds_for_output_interface(interface, self.exclusive_nodes)
        return interface["model"](proxy_inds=tvb_nodes_inds,
                                  voi=voi_inds, voi_labels=voi_labels,
                                  monitor_ind=interface.get("monitor_ind", 0),
                                  spikeNet_receiver_proxy= \
                                    self._build_tvb_to_spikeNet_interface_proxy_nodes(
                                        interface, tvb_nodes_inds, spiking_proxy_inds),
                                  transformer=interface["transformer"],
                                  spiking_network=self.spiking_network,
                                  populations=np.array(interface["populations"]),
                                  spiking_proxy_inds=spiking_proxy_inds)

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        spiking_proxy_inds = self._get_spiking_proxy_inds_for_input_interface(interface, self.exclusive_nodes)
        return interface["model"](proxy_inds=spiking_proxy_inds,
                                  voi=voi_inds, voi_labels=voi_labels,
                                  transformer=interface["transformer"],
                                  spikeNet_sender_proxy= \
                                    self._build_spikeNet_to_tvb_interface_proxy_nodes(interface,
                                                                                      spiking_proxy_inds),
                                  spiking_network=self.spiking_network,
                                  populations=np.array(interface["populations"]),
                                  spiking_proxy_inds=spiking_proxy_inds)

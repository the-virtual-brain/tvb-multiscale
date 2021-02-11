# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


import numpy as np

from tvb.basic.neotraits._attr import HasTraits, Attr, NArray
from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.interfaces.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.io import RemoteSenders, RemoteReceivers, \
    SpikeNetInputDevice, SpikeNetEventsFromOutpuDevice
from tvb_multiscale.core.tvb.interfaces.transformers import TVBTransformers
from tvb_multiscale.core.tvb.interfaces.interfaces import TVBOutputInterfaces, TVBInputInterfaces, \
    TVBSenderInterface, TVBReceiverInterface, TVBTransformerSenderInterface, TVBReceiverTransformerInterface, \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBtoSpikeNetInterfaces, SpikeNetToTVBInterfaces
from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


LOG = initialize_logger(__name__)


class TVBInterfaceBuilder(InterfaceBuilder):
    __metaclass__ = ABCMeta

    """TVBInterfaceBuilder abstract base class"""

    _tvb_output_interfaces_class = TVBOutputInterfaces
    _tvb_input_interfaces_class = TVBInputInterfaces

    tvb_simulator = Attr(label="TVB simulator",
                         doc="""The instance of TVB simulator""",
                         field_type=CoSimulator,
                         required=True)

    exclusive_nodes = Attr(label="TVB simulator",
                           doc="""The instance of TVB simulator""",
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
        self.tvb_simulator.tvb_output_interfaces = self._tvb_output_interfaces_class(interfaces=self._output_interfaces)
        self.tvb_simulator.tvb_input_interfaces = self._tvb_input_interfaces_class(interfaces=self._input_interfaces)
        return self.tvb_simulator


class TVBRemoteInterfaceBuilder(TVBInterfaceBuilder):

    """TVBRemoteInterfaceBuilder class"""

    _remote_senders_types = tuple([val.value for val in RemoteSenders.__members__.values()])
    _remote_receivers_types = tuple([val.value for val in RemoteReceivers.__members__.values()])

    def configure(self):
        super(TVBRemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._remote_senders_types, "sender")
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

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

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

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

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

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

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


class TVBSpikeNetProxyNodesBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """TVBSpikeNetProxyNodesBuilder abstract base class"""

    global_coupling_scaling = NArray(
        dtype=np.float,
        label="Global coupling scaling",
        doc="""Array of global coupling scaling parameters per receiving brain region node.""",
        required=True,
    )

    spiking_network = Attr(label="Spiking Network",
                           doc="""The instance of SpikingNetwork class""",
                           field_type=SpikingNetwork,
                           required=True)

    @property
    @abstractmethod
    def tvb_nodes_inds(self):
        pass

    @property
    @abstractmethod
    def spiking_nodes_inds(self):
        pass

    @property
    @abstractmethod
    def tvb_model_state_variables(self):
        pass

    @property
    @abstractmethod
    def tvb_model_cvar(self):
        pass

    @property
    @abstractmethod
    def number_of_regions(self):
        pass

    @property
    @abstractmethod
    def region_labels(self):
        pass

    @property
    @abstractmethod
    def tvb_coupling_a(self):
        pass

    @property
    @abstractmethod
    def tvb_weights(self):
        pass

    @property
    @abstractmethod
    def tvb_delays(self):
        pass

    def _default_tvb_weight_fun(self, source_node, target_node):
        return self.global_coupling_scaling * self.tvb_weights[source_node, target_node]

    def _default_tvb_delay_fun(self, source_node, target_node):
        return self.tvb_delays[source_node, target_node]

    @property
    @abstractmethod
    def _default_receptor_type(self):
        pass

    @property
    @abstractmethod
    def _default_min_delay(self):
        pass

    def _configure_global_coupling_scaling(self):
        if self.global_coupling_scaling is None:
            self.global_coupling_scaling = self.tvb_coupling_a
        if self.global_coupling_scaling.shape[0] == 1:
            self.global_coupling_scaling = np.repeat(self.global_coupling_scaling, self.number_of_regions, axis=0)
        assert self.global_coupling_scaling.shape[0] == self.number_of_regions

    def configure(self):
        self._configure_global_coupling_scaling()
        super(TVBSpikeNetProxyNodesBuilder, self).configure()

    @property
    @abstractmethod
    def _build_and_connect_devices(self):
        pass

    def _build_tvb_to_spikeNet_interface_proxy_nodes(self, interface, tvb_nodes_inds, spiking_proxy_inds):
        weight_fun = property_to_fun(interface.pop("weights", self._default_tvb_weight_fun))
        delay_fun = property_to_fun(interface.pop("delays", self._default_tvb_delay_fun))
        receptor_type_fun = property_to_fun(interface.get("receptor_type", self._default_receptor_type))
        # Default behavior for any combination of region nodes and populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        # Defaults just follow TVB connectivity
        shape = (len(interface.proxy_inds), len(interface.spiking_proxy_inds.shape[0]))
        weights = np.empty(shape).astype("O")
        delays = np.empty(shape).astype("O")
        receptor_type = np.empty(shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        device_names = []
        # Apply now possible functions per source and target region node:
        for src_node in tvb_nodes_inds:
            i_src = np.where(tvb_nodes_inds == src_node)[0][0]
            device_names.append(self.regions_labels[src_node])
            for i_trg, trg_node in enumerate(spiking_proxy_inds):
                weights[i_src, i_trg] = weight_fun(src_node, trg_node)
                delays[i_src, i_trg] = delay_fun(src_node, trg_node)
                receptor_type[i_src, i_trg] = receptor_type_fun(src_node, trg_node)
                if neurons_inds_fun is not None:
                    neurons_inds[i_src, i_trg] = lambda neurons_inds: neurons_inds_fun(src_node, trg_node, neurons_inds)
        _interface = dict()
        _interface["names"] = device_names
        _interface["weights"] = weights
        _interface["delays"] = delays
        _interface["receptor_type"] = receptor_type
        _interface["neurons_inds"] = neurons_inds
        _interface["nodes"] = [np.where(spiking_proxy_inds == trg_node)[0][0] for trg_node in spiking_proxy_inds]
        # Generate the devices => "proxy TVB nodes":
        return self._build_and_connect_devices([_interface], self.spiking_network.brain_regions), \
               tvb_nodes_inds, spiking_proxy_inds

    def _build_spikeNet_to_tvb_interface_proxy_nodes(self, interface, spiking_proxy_inds):
        delay_fun = property_to_fun(interface.pop("delays", self._default_min_delay))
        # Default behavior for any region node and any combination of populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        shape = (len(interface.spiking_proxy_inds),)
        delays = np.zeros(shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        for i_node, spiking_node in enumerate(spiking_proxy_inds):
            delays[i_node] = delay_fun(spiking_node)
            if neurons_inds_fun is not None:
                neurons_inds[i_node] = lambda neurons_inds: neurons_inds_fun(spiking_node, neurons_inds)
        _interface = dict()
        _interface["delays"] = delays
        _interface["neurons_inds"] = neurons_inds
        # Convert TVB node index to interface SpikeNet node index:
        _interface["nodes"] = [np.where(spiking_proxy_inds == spiking_node)[0][0]
                               for spiking_node in spiking_proxy_inds]
        # Generate the devices <== "proxy TVB nodes":
        return self._build_and_connect_devices([_interface], self.spiking_network.brain_regions)


class TVBSpikeNetInterfaceBuilder(TVBInterfaceBuilder, TVBSpikeNetProxyNodesBuilder):
    __metaclass__ = ABCMeta

    """TVBSpikeNetInterfaceBuilder abstract base class"""

    _tvb_output_interfaces_class = TVBtoSpikeNetInterfaces
    _tvb_input_interfaces_class = SpikeNetToTVBInterfaces

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

    _tvb_to_spikeNet_senders = (SpikeNetInputDevice, )
    _spikeNet_to_tvb_receivers = (SpikeNetEventsFromOutpuDevice, )

    @property
    def tvb_nodes_inds(self):
        return self._default_out_proxy_inds

    @property
    def spiking_nodes_inds(self):
        return self.in_proxy_inds

    def configure(self):
        TVBSpikeNetProxyNodesBuilder.configure(self)
        TVBInterfaceBuilder.configure(self)
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "transformer")
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "transformer")

    def _get_tvb_nodes_spiking_proxy_inds_for_output_interface(self, interface):
        tvb_nodes_inds = self._only_inds(interface.get("proxy_inds", self.tvb_nodes_inds),
                                         self.region_labels)
        spiking_proxy_inds = self._only_inds(interface["spiking_proxy_inds"], self.region_labels)
        if self.exclusive_nodes:
            # TODO: decide about the following:
            #  can a TVB node be updated from a SpikeNet node via a SpikeNet -> TVB interface?,
            #  and get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a direct coupling of that SpikeNet node with other SpikeNet nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in spiking_proxy_inds)
            assert np.all(node not in self.spiking_nodes_ids for node in tvb_nodes_inds)
        return tvb_nodes_inds, spiking_proxy_inds

    def _get_spiking_proxy_inds_for_input_interface(self, interface):
        spiking_proxy_inds = self._only_inds(interface.get("proxy_inds",
                                                           interface.get("spiking_proxy_inds",
                                                                         self.proxy_inds)), self.region_labels)
        if self.exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a NEST node via a NEST -> TVB interface,
            # get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a directly coupling of that NEST node with other NEST nodes?
            assert np.all(spiking_node not in self.tvb_nodes_ids for spiking_node in spiking_proxy_inds)
        return spiking_proxy_inds

    def build_output_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        tvb_nodes_inds, spiking_proxy_inds = self._get_tvb_nodes_spiking_proxy_inds_for_output_interface(interface)
        interface["sender"].target = \
            self._build_tvb_to_spikeNet_interface_proxy_nodes(interface, tvb_nodes_inds, spiking_proxy_inds)
        return TVBtoSpikeNetInterface(proxy_inds=tvb_nodes_inds,
                                      voi=voi_inds, voi_labels=voi_labels,
                                      communicator=interface["sender"],
                                      monitor_ind=interface.get("monitor_ind", 0),
                                      transformer=interface["transformer"],
                                      spiking_network=self.spiking_network,
                                      populations=np.array(interface["populations"]),
                                      spiking_proxy_inds=spiking_proxy_inds)

    def build_input_interface(self, interface):
        voi_inds, voi_labels = self._voi_inds_labels_for_interface(interface)
        spiking_proxy_inds = self._get_spiking_proxy_inds_for_input_interface(interface)
        interface["receiver"].source = self._build_spikeNet_to_tvb_interface_proxy_nodes(interface, spiking_proxy_inds)
        return SpikeNetToTVBInterface(proxy_inds=spiking_proxy_inds,
                                      voi=voi_inds, voi_labels=voi_labels,
                                      communicator=interface["receiver"],
                                      transformer=interface["transformer"],
                                      spiking_network=self.spiking_network,
                                      populations=np.array(interface["populations"]),
                                      spiking_proxy_inds=spiking_proxy_inds)

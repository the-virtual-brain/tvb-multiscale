# -*- coding: utf-8 -*-

from six import string_types
from abc import ABCMeta, abstractmethod, ABC
from enum import Enum

import numpy as np

from tvb.basic.neotraits._attr import NArray, Attr, Float
from tvb.basic.neotraits._core import HasTraits
from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun

from tvb_multiscale.core.interfaces.base.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.base.transformers.models.models import Transformers
from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers

from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetSenderInterface, SpikeNetReceiverInterface, \
    SpikeNetTransformerSenderInterface, SpikeNetReceiverTransformerInterface
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.utils.data_structures_utils import get_enum_values


class DefaultTVBtoSpikeNetModels(Enum):
    RATE = "RATE_TO_SPIKES"
    SPIKES = "SPIKES"
    CURRENT = "CURRENT"


class DefaultSpikeNetToTVBModels(Enum):
    SPIKES = "SPIKES_MEAN"


class SpikeNetProxyNodesBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """SpikeNetProxyNodesBuilder abstract base class"""

    global_coupling_scaling = NArray(
        dtype=np.float,
        label="Global coupling scaling",
        doc="""Array of global coupling scaling parameters per receiving brain region node.""",
        required=True,
        default=np.array([], dtype="f")
    )

    spiking_network = Attr(label="Spiking Network",
                           doc="""The instance of SpikingNetwork class""",
                           field_type=SpikingNetwork,
                           required=True)

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True, default=0.0)

    _tvb_delays = None

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
    def tvb_dt(self):
        pass

    @property
    def spiking_dt(self):
        return self.spiking_network.dt

    @property
    def min_delay(self):
        return self.spiking_network.min_delay

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

    @abstractmethod
    def _get_tvb_delays(self):
        pass

    @property
    def tvb_delays(self):
        if self._tvb_delays is None:
            self._tvb_delays = self._get_tvb_delays()
        return self._tvb_delays

    @property
    @abstractmethod
    def region_labels(self):
        pass

    def _default_tvb_weight_fun(self, source_node, target_node, weights=None):
        if weights is None:
            weights = self.tvb_weights
        return self.global_coupling_scaling[target_node] * weights[source_node, target_node]

    def _default_tvb_delay_fun(self, source_node, target_node, delays=None):
        if delays is None:
            delays = self.tvb_delays
        return delays[source_node, target_node]

    @abstractmethod
    def _default_receptor_type(self, source_node, target_node):
        pass

    @property
    @abstractmethod
    def _default_min_delay(self):
        pass

    @staticmethod
    def _configure_proxy_models(interfaces, interface_models, default_proxy_models, proxy_models):
        # Return a type
        for interface in interfaces:
            model = interface.get("proxy", interface.pop("proxy_model", None))
            if model is None:
                model = interface.get("model", interface_models[0])
                model = model.upper()
                assert model in interface_models
                model = getattr(default_proxy_models, model).value
            if isinstance(model, string_types):
                # string input -> return type
                model = model.upper()
                model = getattr(proxy_models, model).value
            elif isinstance(model, Enum):
                # Enum input -> return type:
                model = model.value
            else:
                # type input
                assert model in get_enum_values(proxy_models)
            interface["proxy"] = model

    def _configure_global_coupling_scaling(self):
        if self.global_coupling_scaling.size == 0:
            self.global_coupling_scaling = self.tvb_coupling_a
        if self.global_coupling_scaling.size == 1:
            self.global_coupling_scaling = np.repeat(self.global_coupling_scaling, self.number_of_regions, axis=0)
        assert self.global_coupling_scaling.shape[0] == self.global_coupling_scaling.size == self.number_of_regions

    def configure(self):
        if self.dt == 0.0:
            self.dt = self.tvb_dt
        super(SpikeNetProxyNodesBuilder, self).configure()
        self._configure_global_coupling_scaling()

    @abstractmethod
    def _build_and_connect_devices(self, interface, **kwargs):
        pass

    def _get_spiking_proxy_inds_for_output_interface(self, interface, exclusive_nodes):
        interface["proxy_inds"] = np.array(self._only_inds(interface.get("proxy_inds", self.tvb_nodes_inds),
                                                  self.region_labels))
        interface["spiking_proxy_inds"] = \
            np.array(self._only_inds(interface.get("spiking_proxy_inds", self.spiking_nodes_inds), self.region_labels))
        if exclusive_nodes:
            # TODO: decide about the following:
            #  can a TVB node be updated from a SpikeNet node via a SpikeNet -> TVB interface?,
            #  and get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a direct coupling of that SpikeNet node with other SpikeNet nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in interface["spiking_proxy_inds"])
            assert np.all(node not in self.spiking_nodes_ids for node in interface["proxy_inds"])

    def _get_spiking_proxy_inds_for_input_interface(self, interface, exclusive_nodes):
        interface["spiking_proxy_inds"] = \
            np.array(self._only_inds(interface.get("proxy_inds",
                                          interface.get("spiking_proxy_inds", self.proxy_inds)), self.region_labels))
        if exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a Spiking Network node
            #  via a SpikeNet -> TVB interface,
            # get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a direct coupling of that SpikeNet node with other SpikeNet nodes?
            assert np.all(spiking_node not in self.tvb_nodes_ids for spiking_node in interface["spiking_proxy_inds"])

    def _build_tvb_to_spikeNet_interface_proxy_nodes(self, interface):
        weight_fun = property_to_fun(interface.pop("weights", self._default_tvb_weight_fun))
        delay_fun = property_to_fun(interface.pop("delays", self._default_tvb_delay_fun))
        receptor_type_fun = property_to_fun(interface.pop("receptor_type", self._default_receptor_type))
        syn_spec_fun = property_to_fun(interface.pop("syn_spec", None))
        conn_spec_fun = property_to_fun(interface.pop("conn_spec", None))
        # Default behavior for any combination of region nodes and populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        # Defaults just follow TVB connectivity
        shape = (len(interface["proxy_inds"]), len(interface["spiking_proxy_inds"]))
        weights = np.empty(shape).astype("O")
        delays = np.empty(shape).astype("O")
        receptor_type = np.empty(shape).astype("O")
        syn_spec = np.tile([None], shape).astype("O")
        conn_spec = np.tile([None], shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        device_names = []
        # Apply now possible functions per source and target region node:
        for src_node in interface["proxy_inds"]:
            i_src = np.where(interface["proxy_inds"] == src_node)[0][0]
            device_names.append(self.region_labels[src_node])
            for i_trg, trg_node in enumerate(interface["spiking_proxy_inds"]):
                weights[i_src, i_trg] = weight_fun(src_node, trg_node, self.tvb_weights)
                delays[i_src, i_trg] = delay_fun(src_node, trg_node, self.tvb_delays)
                receptor_type[i_src, i_trg] = receptor_type_fun(src_node, trg_node)
                syn_spec[i_src, i_trg] = syn_spec_fun(src_node, trg_node)
                conn_spec[i_src, i_trg] = conn_spec_fun(src_node, trg_node)
                if neurons_inds_fun is not None:
                    neurons_inds[i_src, i_trg] = lambda neurons_inds: neurons_inds_fun(src_node, trg_node, neurons_inds)
        _interface = dict()
        _interface["names"] = device_names
        _interface["weights"] = weights
        _interface["delays"] = delays
        _interface["receptor_type"] = receptor_type
        _interface["syn_spec"] = syn_spec
        _interface["conn_spec"] = conn_spec
        _interface["neurons_inds"] = neurons_inds
        _interface["nodes"] = [np.where(self.spiking_nodes_inds == trg_node)[0][0]
                               for trg_node in interface["spiking_proxy_inds"]]
        _interface["model"] = interface["proxy"].model
        _interface["params"] = interface.pop("proxy_params", {})
        # TODO: Figure out if we ever going to need interfaces for multiple state variables!
        _interface["connections"] = {interface["voi_labels"][0].item(): interface["populations"]}
        # Generate the devices => "proxy TVB nodes":
        interface["proxy"] = \
            interface["proxy"](dt=self.dt,
                               target=self._build_and_connect_devices(_interface,
                                                                      devices=self.spiking_network.input_proxies)[0])

    def _build_spikeNet_to_tvb_interface_proxy_nodes(self, interface):
        delay_fun = property_to_fun(interface.pop("delays", self._default_min_delay))
        # Default behavior for any region node and any combination of populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        shape = (len(interface["spiking_proxy_inds"]),)
        delays = np.zeros(shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        for i_node, spiking_node in enumerate(interface["spiking_proxy_inds"]):
            delays[i_node] = delay_fun(spiking_node)
            if neurons_inds_fun is not None:
                neurons_inds[i_node] = lambda neurons_inds: neurons_inds_fun(spiking_node, neurons_inds)
        _interface = dict()
        _interface["delays"] = delays
        _interface["neurons_inds"] = neurons_inds
        # Convert TVB node index to interface SpikeNet node index:
        _interface["nodes"] = [np.where(self.spiking_nodes_inds == spiking_node)[0][0]
                               for spiking_node in interface["spiking_proxy_inds"]]
        _interface["model"] = interface["proxy"].model
        _interface["params"] = interface.pop("proxy_params", {})
        # TODO: Figure out if we ever going to need interfaces for multiple state variables!
        _interface["connections"] = {interface["voi_labels"][0].item(): interface["populations"]}
        # Generate the devices <== "proxy TVB nodes":
        interface["proxy"] = interface["proxy"](dt=self.dt,
                                                source=self._build_and_connect_devices(_interface)[0])


class SpikeNetInterfaceBuilder(InterfaceBuilder, SpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetInterfaceBuilder abstract base class"""

    _tvb_to_spikeNet_models = list(TVBtoSpikeNetModels.__members__)
    _spikeNet_to_TVB_models = list(SpikeNetToTVBModels.__members__)

    _default_tvb_to_spikeNet_models = DefaultTVBtoSpikeNetModels
    _default_spikeNet_to_tvb_models = DefaultSpikeNetToTVBModels

    _input_proxy_models = None
    _output_proxy_models = None

    _output_interface_type = SpikeNetOutputInterface
    _input_interface_type = SpikeNetInputInterface

    tvb_simulator_serialized = Attr(label="TVB simulator serialized",
                                    doc="""Dictionary of TVB simulator serialization""",
                                    field_type=dict,
                                    required=True,
                                    default={})

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                  if the spiking brain regions nodes are modelled exclusively by the spiking simulator, 
                                  i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    @property
    def tvb_dt(self):
        return self.tvb_simulator_serialized.get("integrator.dt", self.config.DEFAULT_DT)

    @property
    def synchronization_time(self):
        return self.tvb_simulator_serialized.get("synchronization_time", 0.0)

    @property
    def synchronization_n_step(self):
        return self.tvb_simulator_serialized.get("synchronization_n_step", 0)

    @property
    def tvb_nsig(self):
        return self.tvb_simulator_serialized.get("integrator.noise.nsig", np.array([0.0]))

    @property
    def tvb_model(self):
        return self.tvb_simulator_serialized.get("model", "")

    @property
    def tvb_model_state_variables(self):
        return self.tvb_simulator_serialized.get("model.state_variables", [])

    @property
    def tvb_model_cvar(self):
        return self.tvb_simulator_serialized.get("model.cvar", [])

    @property
    def number_of_regions(self):
        return self.tvb_simulator_serialized.get("connectivity.number_of_regions", 0)

    @property
    def region_labels(self):
        return self.tvb_simulator_serialized.get("connectivity.region_labels", np.array([]))

    @property
    def tvb_coupling_a(self):
        return self.tvb_simulator_serialized.get("coupling.a", np.array([1.0/256]))

    @property
    def tvb_weights(self):
        return self.tvb_simulator_serialized.get("connectivity.weights", np.zeros((1, 1)))

    def _get_tvb_delays(self):
        # This is good for ANNarchy because one can set the devices' state at time 0.0
        # For NEST, one has to subtract 1 NEST time step.
        return np.maximum(1,
                          self.tvb_simulator_serialized.get("connectivity.idelays", np.ones((1, 1)))
                          - self.synchronization_n_step + 1) * self.tvb_dt

    def _proxy_inds(self, interfaces):
        return np.unique(self._only_inds_for_interfaces(interfaces, "proxy_inds", self.region_labels))

    @property
    def out_proxy_inds(self):
        return self._proxy_inds(self.output_interfaces)

    @property
    def in_proxy_inds(self):
        return self._proxy_inds(self.input_interfaces)

    def region_label_to_ind(self, labels):
        return self._label_to_ind(labels, self.region_labels)

    @property
    def out_proxy_labels(self):
        return self.region_labels[self.out_proxy_inds]

    @property
    def in_proxy_labels(self):
        return self.region_labels[self.in_proxy_inds]

    @property
    def spiking_nodes_inds(self):
        return np.unique(self._proxy_inds(self.output_interfaces).tolist() +
                         self._proxy_inds(self.input_interfaces).tolist())

    @property
    def tvb_nodes_inds(self):
        return self._default_tvb_out_proxy_inds

    def configure(self):
        InterfaceBuilder.configure(self)
        SpikeNetProxyNodesBuilder.configure(self)
        self._configure_proxy_models(self.input_interfaces, self._tvb_to_spikeNet_models,
                                     self._default_tvb_to_spikeNet_models, self._input_proxy_models)
        self._configure_proxy_models(self.output_interfaces,
                                     self._default_spikeNet_to_tvb_models, self._output_proxy_models)
        if len(self.output_interfaces):
            assert self.out_proxy_labels in self.region_labels
        if len(self.input_interfaces):
            assert self.in_proxy_labels in self.region_labels
        self._default_tvb_out_proxy_inds = np.arange(self.number_of_regions).astype('i').tolist()
        if self.exclusive_nodes:
            for proxy_ind in self.spiking_nodes_inds:
                self._default_tvb_out_proxy_inds.remove(proxy_ind)
        self._default_tvb_out_proxy_inds = np.array(self._default_tvb_out_proxy_inds)

    def _get_interface_arguments(self, interface):
        interface.update({"spiking_network": self.spiking_network,
                          "populations": np.array(interface["populations"])})

    def _get_output_interface_arguments(self, interface):
        self._get_interface_arguments(interface)
        interface["dt"] = self.dt
        self._get_spiking_proxy_inds_for_input_interface(interface, self.exclusive_nodes)
        self._build_spikeNet_to_tvb_interface_proxy_nodes(interface)
        return interface

    def _get_input_interface_arguments(self, interface):
        self._get_interface_arguments(interface)
        self._get_spiking_proxy_inds_for_output_interface(interface, self.exclusive_nodes)
        self._build_tvb_to_spikeNet_interface_proxy_nodes(interface)
        return interface

    def build(self):
        self.build_interfaces()
        return self._spikeNet_output_interfaces_type(interfaces=self._output_interfaces), \
               self._spikeNet_intput_interfaces_type(interfaces=self._input_interfaces)


class SpikeNetRemoteInterfaceBuilder(SpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetRemoteInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetSenderInterface
    _input_interface_type = SpikeNetReceiverInterface

    _remote_sender_types = [val.value for val in RemoteSenders.__members__.values()]
    _remote_receiver_types = [val.value for val in RemoteReceivers.__members__.values()]

    def configure(self):
        super(SpikeNetRemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._remote_sender_types, "sender")
        self._assert_input_interfaces_component_config(self._remote_receiver_types, "receiver")

    def _get_output_interface_arguments(self, interface):
        interface = super(SpikeNetRemoteInterfaceBuilder, self)._get_output_interface_arguments(interface)
        interface["communicator"] = interface.pop("sender")
        return interface

    def _get_input_interface_arguments(self, interface):
        interface = super(SpikeNetRemoteInterfaceBuilder, self)._get_input_interface_arguments(interface)
        interface["communicator"] = interface.pop("receiver")
        return interface


class SpikeNetTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder,
                                          TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetTransformerSenderInterface
    _input_interface_type = SpikeNetReceiverTransformerInterface

    _transformer_types = [val.value for val in Transformers.__members__.values()]

    def configure(self):
        super(SpikeNetTransformerInterfaceBuilder, self).configure()
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformer(self, self.input_interfaces)
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformer(self, self.output_interfaces)

    def _get_output_interface_arguments(self, interface):
        return super(SpikeNetTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface)

    def _get_input_interface_arguments(self, interface):
        return super(SpikeNetTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface)


class SpikeNetOutputTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder, SpikeNetToTVBTransformerBuilder, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetOutputTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetTransformerSenderInterface
    _input_interface_type = SpikeNetReceiverInterface

    def configure(self):
        super(SpikeNetOutputTransformerInterfaceBuilder, self).configure()
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformer(self, self.output_interfaces)

    def _get_output_interface_arguments(self, interface):
        return super(SpikeNetOutputTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface)


class SpikeNetInputTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder, TVBtoSpikeNetTransformerBuilder, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetInputTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetSenderInterface
    _input_interface_type = SpikeNetReceiverTransformerInterface

    def configure(self):
        super(SpikeNetInputTransformerInterfaceBuilder, self).configure()
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformer(self, self.input_interfaces)

    def _get_input_interface_arguments(self, interface):
        return super(SpikeNetInputTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface)
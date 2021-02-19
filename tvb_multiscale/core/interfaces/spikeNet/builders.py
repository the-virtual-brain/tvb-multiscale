from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import NArray, Attr, Float
from tvb.basic.neotraits._core import HasTraits
from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun

from tvb_multiscale.core.interfaces.base.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.base.transformers import Transformers
from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetOutputRemoteInterfaces, SpikeNetInputRemoteInterfaces, \
    SpikeNetOutputInterface, SpikeNetInputInterface, \
    SpikeNetSenderInterface, SpikeNetReceiverInterface, \
    SpikeNetTransformerSenderInterface, SpikeNetReceiverTransformerInterface
from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class SpikeNetProxyNodesBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """SpikeNetProxyNodesBuilder abstract base class"""

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

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

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

    @abstractmethod
    def _default_receptor_type(self, source_node, target_node):
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
        super(SpikeNetProxyNodesBuilder, self).configure()

    @abstractmethod
    def _build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        pass

    def _get_tvb_nodes_spiking_proxy_inds_for_output_interface(self, interface, exclusive_nodes):
        tvb_nodes_inds = self._only_inds(interface.get("proxy_inds", self.tvb_nodes_inds),
                                         self.region_labels)
        spiking_proxy_inds = self._only_inds(interface["spiking_proxy_inds"], self.region_labels)
        if exclusive_nodes:
            # TODO: decide about the following:
            #  can a TVB node be updated from a SpikeNet node via a SpikeNet -> TVB interface?,
            #  and get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a direct coupling of that SpikeNet node with other SpikeNet nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in spiking_proxy_inds)
            assert np.all(node not in self.spiking_nodes_ids for node in tvb_nodes_inds)
        return tvb_nodes_inds, spiking_proxy_inds

    def _get_spiking_proxy_inds_for_input_interface(self, interface, exclusive_nodes):
        spiking_proxy_inds = self._only_inds(interface.get("proxy_inds",
                                                           interface.get("spiking_proxy_inds",
                                                                         self.proxy_inds)), self.region_labels)
        if exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a NEST node via a NEST -> TVB interface,
            # get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a directly coupling of that NEST node with other NEST nodes?
            assert np.all(spiking_node not in self.tvb_nodes_ids for spiking_node in spiking_proxy_inds)
        return spiking_proxy_inds

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
        _interface["model"] = self._spikeNet_input_proxy_type.model
        _interface["model"] = interface["proxy_model"].model
        _interface["params"] = interface["proxy_params"]
        # Generate the devices => "proxy TVB nodes":
        return interface["proxy_model"](dt=self.dt,
                                        target=self._build_and_connect_devices([_interface],
                                                                               self.spiking_network.brain_regions))

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
        _interface["model"] = interface["proxy_model"].model
        _interface["params"] = interface["proxy_params"]
        # Generate the devices <== "proxy TVB nodes":
        return interface["proxy_model"](source=self._build_and_connect_devices([_interface],
                                                                               self.spiking_network.brain_regions))


class SpikeNetInterfaceBuilder(InterfaceBuilder, SpikeNetProxyNodesBuilder):
    __metaclass__ = ABCMeta

    """SpikeNetInterfaceBuilder abstract base class"""

    _output_interfaces_type = SpikeNetOutputRemoteInterfaces
    _input_interfaces_type = SpikeNetInputRemoteInterfaces

    _output_interface_type = SpikeNetOutputInterface
    _input_interface_type = SpikeNetInputInterface

    tvb_simulator_serialized = Attr(label="TVB simulator serialized",
                                    doc="""Dictionary of TVB simulator serialization""",
                                    field_type=dict,
                                    required=True)

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                  if the spiking brain regions nodes are modelled exclusively by the spiking simulator, 
                                  i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    @property
    def tvb_dt(self):
        return self.tvb_simulator_serialized["integrator.dt"]

    @property
    def tvb_model(self):
        return self.tvb_simulator_serialized["model"]

    @property
    def tvb_model_state_variables(self):
        return self.tvb_simulator_serialized["model.state_variables"]

    @property
    def tvb_model_cvar(self):
        return self.tvb_simulator_serialized["model.cvar"]

    @property
    def number_of_regions(self):
        return self.tvb_simulator_serialized["connectivity.number_of_regions"]

    @property
    def region_labels(self):
        return self.tvb_simulator_serialized["connectivity.region_labels"]

    @property
    def tvb_coupling_a(self):
        return self.tvb_simulator_serialized["coupling.a"]

    @property
    def tvb_weights(self):
        return self.tvb_simulator_serialized["connectivity.weights"]

    @property
    def tvb_delays(self):
        return self.tvb_simulator_serialized["connectivity.delays"]

    def _proxy_inds(self, interfaces):
        return np.unique(self._only_inds_for_interfaces(interfaces, "proxy", self.region_labels))

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
        self.dt = self.tvb_dt  # From SpikeNetInterfaceBuilder to SpikeNetProxyNodesBuilder
        SpikeNetProxyNodesBuilder.configure(self)
        InterfaceBuilder.configure(self)
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
        return {"spiking_network": self.spiking_network,
                "populations": np.array(interface["populations"])}

    def _get_output_interface_arguments(self, interface):
        kwargs = self._get_interface_arguments(interface)
        kwargs["spiking_proxy_inds"] = self._get_spiking_proxy_inds_for_input_interface(interface, self.exclusive_nodes)
        kwargs["spikeNet_sender_proxy"] = \
            self._build_spikeNet_to_tvb_interface_proxy_nodes(interface, kwargs["spiking_proxy_inds"])
        return kwargs

    def _get_input_interface_arguments(self, interface):
        kwargs = self._get_interface_arguments(interface)
        tvb_nodes_inds, spiking_proxy_inds = \
            self._get_tvb_nodes_spiking_proxy_inds_for_output_interface(interface, self.exclusive_nodes)
        kwargs["spiking_proxy_inds"] = spiking_proxy_inds
        kwargs["spikeNet_receiver_proxy"] = \
            self._build_tvb_to_spikeNet_interface_proxy_nodes(interface, tvb_nodes_inds, spiking_proxy_inds)
        return kwargs

    def build(self):
        self.build_interfaces()
        return self._spikeNet_output_interfaces_type(interfaces=self._output_interfaces), \
               self._spikeNet_intput_interfaces_type(interfaces=self._input_interfaces)


class SpikeNetRemoteInterfaceBuilder(SpikeNetInterfaceBuilder):
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
        kwargs = super(SpikeNetRemoteInterfaceBuilder, self)._get_output_interface_arguments(interface)
        kwargs["communicator"] = interface["sender"]
        return kwargs

    def _get_input_interface_arguments(self, interface):
        kwargs = super(SpikeNetRemoteInterfaceBuilder, self)._get_input_interface_arguments(interface)
        kwargs["communicator"] = interface["receiver"]
        return kwargs


class SpikeNetTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder):
    __metaclass__ = ABCMeta

    """SpikeNetTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetTransformerSenderInterface
    _input_interface_type = SpikeNetReceiverTransformerInterface

    _transformer_types = [val.value for val in Transformers.__members__.values()]

    def configure(self):
        super(SpikeNetTransformerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._transformer_types, "transformer")
        self._assert_input_interfaces_component_config(self._transformer_types, "transformer")

    def _get_output_interface_arguments(self, interface):
        kwargs = super(SpikeNetTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface)
        kwargs["transformer"] = interface["transformer"]
        return kwargs

    def _get_input_interface_arguments(self, interface):
        kwargs = super(SpikeNetTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface)
        kwargs["transformer"] = interface["transformer"]
        return kwargs


class SpikeNetOutputTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder):
    __metaclass__ = ABCMeta

    """SpikeNetOutputTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetTransformerSenderInterface
    _input_interface_type = SpikeNetReceiverInterface

    _transformer_types = [val.value for val in Transformers.__members__.values()]

    def configure(self):
        super(SpikeNetOutputTransformerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._transformer_types, "transformer")

    def _get_output_interface_arguments(self, interface):
        kwargs = super(SpikeNetOutputTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface)
        kwargs["transformer"] = interface["transformer"]
        return kwargs


class SpikeNetInputTransformerInterfaceBuilder(SpikeNetRemoteInterfaceBuilder):
    __metaclass__ = ABCMeta

    """SpikeNetInputTransformerInterfaceBuilder abstract base class"""

    _output_interface_type = SpikeNetSenderInterface
    _input_interface_type = SpikeNetReceiverTransformerInterface

    _transformer_types = [val.value for val in Transformers.__members__.values()]

    def configure(self):
        super(SpikeNetInputTransformerInterfaceBuilder, self).configure()
        self._assert_input_interfaces_component_config(self._transformer_types, "transformer")

    def _get_input_interface_arguments(self, interface):
        kwargs = super(SpikeNetInputTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface)
        kwargs["transformer"] = interface["transformer"]
        return kwargs

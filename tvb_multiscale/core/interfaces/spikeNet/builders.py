from abc import ABCMeta, abstractmethod

import numpy as np
from tvb.basic.neotraits._attr import NArray, Attr
from tvb.basic.neotraits._core import HasTraits

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
        super(SpikeNetProxyNodesBuilder, self).configure()

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
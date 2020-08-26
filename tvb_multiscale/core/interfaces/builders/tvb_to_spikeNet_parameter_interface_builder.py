# -*- coding: utf-8 -*-
from six import string_types
from pandas import Series
import numpy as np

from tvb_multiscale.core.config import CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.tvb_to_spikeNet_parameter_interface import TVBtoSpikeNetParameterInterface

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


class TVBtoSpikeNetParameterInterfaceBuilder(object):
    _build_target_class = TVBtoSpikeNetParameterInterface
    interface = []
    spiking_nodes = Series()
    tvb_nodes_ids = []
    spiking_nodes_ids = []
    exclusive_nodes = False
    config = CONFIGURED

    def __init__(self, interfaces, spiking_network, spiking_nodes, spiking_nodes_ids,
                 tvb_nodes_ids, tvb_model, exclusive_nodes=False, config=CONFIGURED):
        self.interfaces = interfaces
        self.spiking_network = spiking_network
        self.spiking_nodes = spiking_nodes
        self.spiking_nodes_ids = spiking_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model
        self.exclusive_nodes = exclusive_nodes
        self.config = config

    def build_interface(self, interface, interface_id):
        # One interface for every combination of Spiking node
        # and TVB state variable to be transmitted
        # from TVB to Spiking Network
        connections = interface["connections"]
        if isinstance(connections, string_types):
            connections = {connections: slice(None)}  # return all population types
        default_parameter = self._build_target_class._available_input_parameters[interface["model"]]
        spiking_nodes_ids = interface.get("nodes", self.spiking_nodes_ids)
        if spiking_nodes_ids is None:
            spiking_nodes_ids = self.spiking_nodes_ids
        spiking_nodes_ids = list(spiking_nodes_ids)
        if self.exclusive_nodes:
            assert np.all(spiking_node not in self.tvb_nodes_ids for spiking_node in spiking_nodes_ids)
        interface_weight_fun = property_to_fun(interface.get("interface_weights", 1.0))
        # Default behavior for any region node and any combination of populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        shape = (len(spiking_nodes_ids),)
        interface_weights = np.ones(shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        for i_node, spiking_node_id in enumerate(spiking_nodes_ids):
            interface_weights[i_node] = interface_weight_fun(spiking_node_id)
            if neurons_inds_fun is not None:
                neurons_inds[i_node] = lambda neurons_inds: neurons_inds_fun(spiking_node_id, neurons_inds)
        tvb_to_spikeNet_interfaces = Series()
        for name, populations in connections.items():
            try:
                tvb_coupling_id = self.tvb_model.cvar.tolist().index(
                    self.tvb_model.state_variables.index(name))
            except:
                raise_value_error("Failed to compute the coupling index of TVB state variable %s!" % name)
            interface_index = "%s_%d" % (name, interface_id)
            tvb_to_spikeNet_interfaces[interface_index] = \
                self._build_target_class(self.spiking_network, name, interface["model"],
                                         interface.get("parameter", default_parameter),
                                         tvb_coupling_id, spiking_nodes_ids, interface_weights)
            for node in self.spiking_nodes:
                tvb_to_spikeNet_interfaces[interface_index][node.label] = node[populations]
            return tvb_to_spikeNet_interfaces

    def build(self):
        tvb_to_spikeNet_interfaces = Series()
        for id, interface in enumerate(self.interfaces):
            tvb_to_spikeNet_interfaces = tvb_to_spikeNet_interfaces.append(self.build_interface(interface, id))
        return tvb_to_spikeNet_interfaces

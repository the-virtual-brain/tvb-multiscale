# -*- coding: utf-8 -*-
from six import string_types
from pandas import Series
import numpy as np
from tvb_nest.interfaces.tvb_to_nest_parameter_interface import TVBNESTParameterInterface, NEST_INPUT_PARAMETERS
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


class TVBtoNESTInterfaceParameterBuilder(object):
    interface = []
    nest_instance = None
    nest_nodes = Series()
    tvb_nodes_ids = []
    nest_nodes_ids = []
    connectivity = None

    def __init__(self, interfaces, nest_instance, nest_nodes, nest_nodes_ids, tvb_nodes_ids, tvb_model):
        self.interfaces = interfaces
        self.nest_instance = nest_instance
        self.nest_nodes = nest_nodes
        self.nest_nodes_ids = nest_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model

    def build_interface(self, interface):
        # One interface for every combination NEST node
        # and TVB state variable to be transmitted
        # from TVB to NEST
        connections = interface["connections"]
        if isinstance(connections, string_types):
            connections = {connections: slice(None)}  # return all population types
        default_parameter = NEST_INPUT_PARAMETERS[interface["model"]]
        nest_nodes_ids = interface.get("nodes", self.nest_nodes_ids)
        if nest_nodes_ids is None:
            nest_nodes_ids = self.nest_nodes_ids
        nest_nodes_ids = list(nest_nodes_ids)
        interface_weights = 1.0 * np.ones((len(nest_nodes_ids),)).astype("f")
        interface_weight = property_to_fun(interface.get("interface_weights", 1.0))
        for i_w, nest_node_id in enumerate(nest_nodes_ids):
            interface_weights[i_w] = interface_weight(nest_node_id)
        tvb_to_nest_interfaces = Series()
        for name, populations in connections.items():
            try:
                tvb_coupling_id = self.tvb_model.cvar.tolist().index(
                    self.tvb_model.state_variables.index(name))
            except:
                raise ValueError("Failed to compute the coupling index of TVB state variable %s!" % name)
            tvb_to_nest_interfaces[name] = TVBNESTParameterInterface(self.nest_instance, name, interface["model"],
                                                         interface.get("parameter", default_parameter),
                                                         tvb_coupling_id, nest_nodes_ids, interface_weights)
            for node in self.nest_nodes:
                tvb_to_nest_interfaces[name][node.label] = node[populations]
            return tvb_to_nest_interfaces

    def build(self):
        tvb_to_nest_interfaces = Series()
        for interface in self.interfaces:
            tvb_to_nest_interfaces = tvb_to_nest_interfaces.append(self.build_interface(interface))
        return tvb_to_nest_interfaces

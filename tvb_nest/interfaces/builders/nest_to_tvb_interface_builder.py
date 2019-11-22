# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_nest.simulator_nest.nest_factory import build_and_connect_output_devices
from tvb_nest.interfaces.nest_to_tvb_interface import NESTtoTVBinterface
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


class NESTtoTVBInterfaceBuilder(object):
    interfaces = []
    nest_instance = None
    nest_nodes = Series()
    nest_nodes_ids = []
    tvb_nodes_ids = []
    tvb_model = None
    connectivity = None

    def __init__(self, interfaces, nest_instance, nest_nodes, nest_nodes_ids, tvb_nodes_ids, tvb_model, connectivity):
        self.interfaces = interfaces
        self.nest_instance = nest_instance
        self.nest_nodes = nest_nodes
        self.nest_nodes_ids = nest_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model
        self.connectivity = connectivity

    def build_interface(self, interface):
        # One NEST output device for every combination of NEST node
        # and TVB state variable/parameter to be transmitted
        # from NEST to TVB
        nest_to_tvb_interface = Series()
        nest_nodes = interface.get("nodes", self.nest_nodes_ids)  # Indices corresponding to NEST nodes
        if nest_nodes is None:
            nest_nodes = self.nest_nodes_ids
        # We prefer to multiply interface_weights outside NEST:
        interface_weights = np.ones((len(nest_nodes),)).astype("f")
        interface_weight = property_to_fun(interface.pop("interface_weights", 1.0))
        delays = np.ones((len(nest_nodes),)).astype("f")
        delay = property_to_fun(interface.pop("delays", 0.0))
        for i_w, nest_node in enumerate(nest_nodes):
            interface_weights[i_w] = interface_weight(nest_node)
            delays[i_w] = delay(nest_node)
        # Delays should be set to the device
        interface["delays"] = delays  # Per node
        # Convert TVB node index to interface NEST node index:
        nest_nodes_ids = [np.where(self.nest_nodes_ids == nest_node)[0][0] for nest_node in nest_nodes]
        interface["nodes"] = nest_nodes_ids
        devices = build_and_connect_output_devices(self.nest_instance, [interface], self.nest_nodes)
        for name, device_set in devices.items():
            try:
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                tvb_sv_id = None  # it might be a TVB parameter, not a state variable
            nest_to_tvb_interface[name] = NESTtoTVBinterface(nodes_ids=nest_nodes,
                                                             interface_weights=interface_weights). \
                                                                        from_device_set(device_set, tvb_sv_id, name)
        return nest_to_tvb_interface

    def build(self):
        nest_to_tvb_interfaces = Series()
        for interface in self.interfaces:
            nest_to_tvb_interfaces = \
                nest_to_tvb_interfaces.append(self.build_interface(interface))
        return nest_to_tvb_interfaces

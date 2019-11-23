# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_nest.simulator_nest.nest_factory import build_and_connect_output_devices
from tvb_nest.interfaces.tvb_to_nest_interface import TVBtoNESTinterface
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


class TVBtoNESTInterfaceDeviceBuilder(object):
    interfaces = []
    nest_instance = None
    nest_nodes = Series()
    tvb_nodes_ids = []
    nest_nodes_ids = []
    tvb_model = None
    connectivity = None
    tvb_dt = 0.1
    exclusive_nodes = False

    def __init__(self, interfaces, nest_instance, nest_nodes, nest_nodes_ids,
                 tvb_nodes_ids, tvb_model, connectivity, tvb_dt, exclusive_nodes=False):
        self.interfaces = interfaces
        self.nest_instance = nest_instance
        self.nest_nodes = nest_nodes
        self.nest_nodes_ids = nest_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model
        self.connectivity = connectivity
        self.tvb_dt = tvb_dt
        self.exclusive_nodes = exclusive_nodes

    @property
    def node_labels(self):
        return self.connectivity.region_labels

    def build_interface(self, interface):
        # One NEST stimulation device for every combination of
        # TVB node and state variable to be transmitted from TVB to NEST
        source_tvb_nodes = interface.pop("source_nodes", self.tvb_nodes_ids)
        if source_tvb_nodes is None:
            source_tvb_nodes = self.tvb_nodes_ids
        source_tvb_nodes = list(source_tvb_nodes)
        target_nest_nodes = interface.pop("target_nodes", self.nest_nodes_ids)
        if target_nest_nodes is None:
            target_nest_nodes = self.nest_nodes_ids
        target_nest_nodes = list(target_nest_nodes)
        if self.exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a NEST node via a NEST -> TVB interface,
            # get simulated in TVB and again update NEST via a TVB -> NEST interface?
            # Will it depend on whether there is also a directly coupling of that NEST node with other NEST nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in target_nest_nodes)
            assert np.all(node not in self.nest_nodes_ids for node in source_tvb_nodes)
        interface_weight = property_to_fun(interface.pop("interface_weights", 1.0))
        interface_weights = np.ones((len(source_tvb_nodes),)).astype("f")
        weight = property_to_fun(interface.pop("weights", 1.0))
        delay = property_to_fun(interface.pop("delays", 0.0))
        receptor_type = property_to_fun(interface.pop("receptor_types", 0))
        # TODO: Find a way to change self directed weights in cases of non exclusive TVB and NEST nodes!
        weights = self.connectivity.weights[source_tvb_nodes][:, target_nest_nodes]
        delays = self.connectivity.delays[source_tvb_nodes][:, target_nest_nodes]
        receptor_types = np.zeros(delays.shape).astype("i")
        target_nest_nodes_ids = [np.where(self.nest_nodes_ids == trg_node)[0][0] for trg_node in target_nest_nodes]
        interface["nodes"] = target_nest_nodes_ids
        device_names = []
        for src_node in source_tvb_nodes:
            i_src = np.where(self.tvb_nodes_ids == src_node)[0][0]
            interface_weights[i_src] = interface_weight(src_node)
            device_names.append(self.node_labels[src_node])
            for trg_node, i_trg in zip(target_nest_nodes, target_nest_nodes_ids):
                weights[i_src, i_trg] *= weight(src_node, trg_node)
                delays[i_src, i_trg] += delay(src_node, trg_node)
                receptor_types[i_src, i_trg] = receptor_type(src_node, trg_node)
        interface["weights"] = weights
        interface["delays"] = delays
        interface["receptor_types"] = receptor_types
        interface["names"] = device_names
        devices = build_and_connect_output_devices(self.nest_instance, [interface], self.nest_nodes)
        tvb_to_nest_interface = Series()
        for name, device in devices.items():
            tvb_to_nest_interface[name] = \
                TVBtoNESTinterface(self.nest_instance, nodes_ids=source_tvb_nodes, target_nodes=target_nest_nodes,
                                   interface_weights=interface_weights, dt=self.tvb_dt). \
                                                            from_device_set(device,
                                                                            self.tvb_model.state_variables.index(name),
                                                                            name)
            if len(source_tvb_nodes) * len(target_nest_nodes) > 0:
                assert np.abs(np.max(tvb_to_nest_interface[name].weights - weights)) < 0.001
                assert np.abs(np.max(tvb_to_nest_interface[name].delays - delays)) < 1.0  # ms
                assert np.abs(np.max(tvb_to_nest_interface[name].receptors - receptor_types)) < 1  # integers
        return tvb_to_nest_interface

    def build(self):
        tvb_to_nest_interfaces = Series()
        for interface in self.interfaces:
            tvb_to_nest_interfaces = tvb_to_nest_interfaces.append(self.build_interface(interface))
        return tvb_to_nest_interfaces
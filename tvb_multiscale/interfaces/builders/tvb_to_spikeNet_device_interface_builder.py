# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from pandas import Series
import numpy as np
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


@add_metaclass(ABCMeta)
class TVBtoSpikeNetDeviceInterfaceBuilder(object):
    _available_input_devices = {}
    interfaces = []
    spiking_nodes = Series()
    tvb_nodes_ids = []
    spiking_nodes_ids = []
    tvb_model = None
    tvb_weights = None
    tvb_delays = None
    tvb_dt = 0.1
    exclusive_nodes = False
    node_labels = None

    def __init__(self, interfaces, spiking_network, spiking_nodes, spiking_nodes_ids,
                 tvb_nodes_ids, tvb_model, tvb_weights, tvb_delays, node_labels, tvb_dt, exclusive_nodes=False):
        self.interfaces = interfaces
        self.spiking_network = spiking_network
        self.spiking_nodes = spiking_nodes
        self.spiking_nodes_ids = spiking_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model
        self.tvb_weights = tvb_weights
        self.tvb_delays = tvb_delays
        self.node_labels = node_labels
        self.tvb_dt = tvb_dt
        self.exclusive_nodes = exclusive_nodes

    @abstractmethod
    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        pass

    def build_interface(self, interface):
        # One NEST stimulation device for every combination of
        # TVB node and state variable to be transmitted from TVB to NEST
        source_tvb_nodes = interface.pop("source_nodes", self.tvb_nodes_ids)
        if source_tvb_nodes is None:
            source_tvb_nodes = self.tvb_nodes_ids
        source_tvb_nodes = list(source_tvb_nodes)
        target_nodes = interface.pop("target_nodes", self.spiking_nodes_ids)
        if target_nodes is None:
            target_nodes = self.spiking_nodes_ids
        target_nodes = list(target_nodes)
        if self.exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a NEST node via a NEST -> TVB interface,
            # get simulated in TVB and again update NEST via a TVB -> NEST interface?
            # Will it depend on whether there is also a directly coupling of that NEST node with other NEST nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in target_nodes)
            assert np.all(node not in self.spiking_nodes_ids for node in source_tvb_nodes)
        interface_weight = property_to_fun(interface.pop("interface_weights", 1.0))
        interface_weights = np.ones((len(source_tvb_nodes),)).astype("f")
        weight = property_to_fun(interface.pop("weights", 1.0))
        delay = property_to_fun(interface.pop("delays", 0.0))
        receptor_type = property_to_fun(interface.pop("receptor_types", 0))
        # TODO: Find a way to change self directed weights in cases of non exclusive TVB and NEST nodes!
        weights = np.array(self.tvb_weights[source_tvb_nodes][:, target_nodes])
        delays = np.array(self.tvb_delays[source_tvb_nodes][:, target_nodes])
        receptor_types = np.zeros(delays.shape).astype("i")
        target_nodes_ids = [np.where(self.spiking_nodes_ids == trg_node)[0][0] for trg_node in target_nodes]
        interface["nodes"] = target_nodes_ids
        device_names = []
        for src_node in source_tvb_nodes:
            i_src = np.where(self.tvb_nodes_ids == src_node)[0][0]
            interface_weights[i_src] = interface_weight(src_node)
            device_names.append(self.node_labels[src_node])
            for trg_node, i_trg in zip(target_nodes, target_nodes_ids):
                weights[i_src, i_trg] = weight(src_node, trg_node, self.tvb_weights)
                delays[i_src, i_trg] = delay(src_node, trg_node, self.tvb_delays)
                receptor_types[i_src, i_trg] = receptor_type(src_node, trg_node)
        interface["weights"] = weights
        interface["delays"] = delays
        interface["receptor_types"] = receptor_types
        interface["names"] = device_names
        devices = self.build_and_connect_devices([interface], self.spiking_nodes)
        tvb_to_spikeNet_interface = Series()
        for name, device in devices.items():
            try:
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                raise_value_error("Interface with %s doesn't correspond to a TVB state variable!")
            try:
                interface_builder = self._available_input_devices[device.model]
            except:
                raise_value_error("Interface model %s is not supported yet!" % device.model)
            tvb_to_spikeNet_interface[name] = \
                interface_builder(self.spiking_network,
                                  nodes_ids=source_tvb_nodes,
                                  target_nodes=target_nodes,
                                  scale=interface_weights,
                                  dt=self.tvb_dt).from_device_set(device, tvb_sv_id, name)
            if len(source_tvb_nodes) * len(target_nodes) > 0:
                # TODO: an assertion check to include dictionaries
                # assert np.abs(np.max(tvb_to_spikeNet_interface[name].weights - weights)) < 0.001
                # assert np.abs(np.max(tvb_to_spikeNet_interface[name].delays - delays)) < 1.0  # ms
                assert np.abs(np.max(tvb_to_spikeNet_interface[name].receptors - receptor_types)) < 1  # integers
        return tvb_to_spikeNet_interface

    def build(self):
        tvb_to_spikeNet_interfaces = Series()
        for interface in self.interfaces:
            tvb_to_spikeNet_interfaces = tvb_to_spikeNet_interfaces.append(self.build_interface(interface))
        return tvb_to_spikeNet_interfaces
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np
from pandas import Series
from six import add_metaclass
from tvb.simulator.plot.utils.data_structures_utils import property_to_fun
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error

from tvb_multiscale.config import CONFIGURED

LOG = initialize_logger(__name__)


@add_metaclass(ABCMeta)
class TVBtoSpikeNetDeviceInterfaceBuilder(object):
    _available_input_device_interfaces = {}
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
    config = CONFIGURED

    def __init__(self, interfaces, spiking_network, spiking_nodes, spiking_nodes_ids,
                 tvb_nodes_ids, tvb_model, tvb_weights, tvb_delays, node_labels, tvb_dt, exclusive_nodes=False,
                 config=CONFIGURED):
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
        self.config = config

    @abstractmethod
    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        pass

    def build_interface(self, interface):
        # One SpikeNet stimulation device for every combination of
        # TVB node and state variable to be transmitted from TVB to SpikeNet
        source_tvb_nodes = interface.pop("source_nodes", self.tvb_nodes_ids)
        if source_tvb_nodes is None:
            source_tvb_nodes = self.tvb_nodes_ids
        source_tvb_nodes = list(source_tvb_nodes)
        target_nodes = interface.pop("target_nodes", self.spiking_nodes_ids)
        if target_nodes is None:
            target_nodes = self.spiking_nodes_ids
        target_nodes = list(target_nodes)
        if self.exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a SpikeNet node via a SpikeNet -> TVB interface,
            # get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a direct coupling of that SpikeNet node with other SpikeNet nodes?
            assert np.all(node not in self.tvb_nodes_ids for node in target_nodes)
            assert np.all(node not in self.spiking_nodes_ids for node in source_tvb_nodes)
        # Properties set as functions
        interface_weight_fun = property_to_fun(interface.pop("interface_weights", 1.0))
        interface_weights = np.ones((len(source_tvb_nodes),)).astype("f")
        weight_fun = property_to_fun(interface.pop("weights", 1.0))
        delay_fun = property_to_fun(interface.pop("delays", 0.0))
        receptor_type_fun = property_to_fun(interface.pop("receptor_types", 0))
        # TODO: Find a way to change self directed weights in cases of non exclusive TVB and Spiking Network nodes!
        # Defaults just follow TVB connectivity
        weights = np.array(self.tvb_weights[source_tvb_nodes][:, target_nodes]).astype("O")
        delays = np.array(self.tvb_delays[source_tvb_nodes][:, target_nodes]).astype("O")
        receptor_types = np.zeros(delays.shape).astype("i")
        target_nodes_ids = [np.where(self.spiking_nodes_ids == trg_node)[0][0] for trg_node in target_nodes]
        interface["nodes"] = target_nodes_ids
        device_names = []
        # Apply now possible functions per source and target region node:
        for src_node in source_tvb_nodes:
            i_src = np.where(self.tvb_nodes_ids == src_node)[0][0]
            interface_weights[i_src] = interface_weight_fun(src_node)
            device_names.append(self.node_labels[src_node])
            for trg_node, i_trg in zip(target_nodes, target_nodes_ids):
                weights[i_src, i_trg] = weight_fun(src_node, trg_node)
                delays[i_src, i_trg] = delay_fun(src_node, trg_node)
                receptor_types[i_src, i_trg] = receptor_type_fun(src_node, trg_node)
        interface["weights"] = weights
        interface["delays"] = delays
        interface["receptor_types"] = receptor_types
        interface["names"] = device_names
        # Generate the devices => "proxy TVB nodes":
        devices = self.build_and_connect_devices([interface], self.spiking_nodes)
        tvb_to_spikeNet_interface = Series()
        for name, device in devices.items():
            try:
                # The TVB state variable index linked to the interface to build
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                raise_value_error("Interface with %s doesn't correspond to a TVB state variable!")
            try:
                interface_builder = self._available_input_device_interfaces[device.model]
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

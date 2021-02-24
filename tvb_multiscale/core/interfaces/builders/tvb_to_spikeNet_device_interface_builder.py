# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from pandas import Series
import numpy as np

from tvb_multiscale.core.config import CONFIGURED, initialize_logger

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun


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

    tvb_to_spiking_dt_ratio = config.TVB_TO_SPIKING_DT_RATIO
    default_min_spiking_dt = config.MIN_SPIKING_DT
    default_min_delay_ratio = config.MIN_DELAY_RATIO
    default_min_delay = config.MIN_SPIKING_DT

    default_connection = {}

    def __init__(self, interfaces, spiking_network, spiking_nodes_ids,
                 tvb_nodes_ids, tvb_model, tvb_weights, tvb_delays, node_labels, tvb_dt, exclusive_nodes=False,
                 config=CONFIGURED):
        self.interfaces = interfaces
        self.spiking_network = spiking_network
        self.spiking_nodes_ids = spiking_nodes_ids
        self.tvb_nodes_ids = tvb_nodes_ids
        self.tvb_model = tvb_model
        self.tvb_weights = tvb_weights
        self.tvb_delays = tvb_delays
        self.node_labels = node_labels
        self.tvb_dt = tvb_dt
        self.exclusive_nodes = exclusive_nodes
        self.config = config
        self._update_default_min_delay()
        self.default_connection = dict(self.config.DEFAULT_CONNECTION)
        self.default_connection["delay"] = self.default_min_delay

    @property
    def spiking_dt(self):
        return int(np.round(self.tvb_dt / self.tvb_to_spiking_dt_ratio / self.default_min_spiking_dt)) \
                          * self.default_min_spiking_dt

    @property
    def min_delay(self):
        return self.default_min_delay

    def _update_default_min_delay(self):
        # The Spiking Network min delay should be smaller than half the TVB dt,
        # and an integer multiple of the spiking simulator dt
        self.default_min_delay = np.minimum(
            np.maximum(self.default_min_delay_ratio * self.spiking_dt, self.min_delay),
            self.tvb_dt / 2)

    @abstractmethod
    def build_and_connect_devices(self, devices, nodes):
        pass

    def build_interface(self, interface, interface_id):
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
        weight_fun = property_to_fun(interface.pop("weights", self.default_connection["weight"]))
        delay_fun = property_to_fun(interface.pop("delays", self.default_connection["delay"]))
        receptor_type_fun = property_to_fun(interface.pop("receptor_type",
                                                          self.default_connection["receptor_type"]))
        syn_spec_fun = property_to_fun(interface.pop("syn_spec", None))
        conn_spec_fun = property_to_fun(interface.pop("conn_spec", None))
        # Default behavior for any combination of region nodes and populations
        # is to target all of their neurons:
        neurons_inds_fun = interface.pop("neurons_inds", None)
        if neurons_inds_fun is not None:
            neurons_inds_fun = property_to_fun(neurons_inds_fun)
        # TODO: Find a way to change self directed weights in cases of non exclusive TVB and Spiking Network nodes!
        # Defaults just follow TVB connectivity
        weights = np.array(self.tvb_weights[source_tvb_nodes][:, target_nodes]).astype("O")
        delays = np.array(self.tvb_delays[source_tvb_nodes][:, target_nodes]).astype("O")
        shape = delays.shape
        receptor_type = np.tile(self.default_connection["receptor_type"], shape).astype("O")
        neurons_inds = np.tile([None], shape).astype("O")
        syn_spec = np.tile([None], shape).astype("O")
        conn_spec = np.tile([None], shape).astype("O")
        device_names = []
        # Apply now possible functions per source and target region node:
        for src_node in source_tvb_nodes:
            i_src = np.where(self.tvb_nodes_ids == src_node)[0][0]
            interface_weights[i_src] = interface_weight_fun(src_node)
            device_names.append(self.node_labels[src_node])
            for i_trg, trg_node in enumerate(target_nodes):
                weights[i_src, i_trg] = weight_fun(src_node, trg_node)
                delays[i_src, i_trg] = delay_fun(src_node, trg_node)
                receptor_type[i_src, i_trg] = receptor_type_fun(src_node, trg_node)
                syn_spec[i_src, i_trg] = syn_spec_fun(src_node, trg_node)
                conn_spec[i_src, i_trg] = conn_spec_fun(src_node, trg_node)
                if neurons_inds_fun is not None:
                    neurons_inds[i_src, i_trg] = lambda neurons_inds: neurons_inds_fun(src_node, trg_node, neurons_inds)
        interface["names"] = device_names
        interface["weights"] = weights
        interface["delays"] = delays
        interface["receptor_type"] = receptor_type
        interface["syn_spec"] = syn_spec
        interface["conn_spec"] = conn_spec
        interface["neurons_inds"] = neurons_inds
        interface["nodes"] = [np.where(self.spiking_nodes_ids == trg_node)[0][0] for trg_node in target_nodes]
        # Generate the devices => "proxy TVB nodes":
        device_sets = self.build_and_connect_devices([interface], self.spiking_network.brain_regions)
        tvb_to_spikeNet_interface = Series()
        for device_set, target_spiking_pops in zip(device_sets, list(interface["connections"].values())):
            try:
                # The TVB state variable index linked to the interface to build
                tvb_sv_id = self.tvb_model.state_variables.index(device_set.name)
            except:
                raise_value_error("Interface with %s doesn't correspond to a TVB state variable!" % device_set.name)
            try:
                interface_builder = self._available_input_device_interfaces[device_set.model]
            except:
                raise_value_error("Interface model %s is not supported yet!" % device_set.model)
            interface_index = "%d_%s->%s" % (interface_id, device_set.name, str(target_spiking_pops))
            tvb_to_spikeNet_interface[interface_index] = \
                interface_builder(self.spiking_network,
                                  nodes_ids=source_tvb_nodes,
                                  target_nodes=target_nodes,
                                  scale=interface_weights,
                                  dt=self.tvb_dt).from_device_set(device_set, tvb_sv_id, device_set.name)
        return tvb_to_spikeNet_interface

    def build(self):
        tvb_to_spikeNet_interfaces = Series()
        for id, interface in enumerate(self.interfaces):
            tvb_to_spikeNet_interfaces = tvb_to_spikeNet_interfaces.append(self.build_interface(interface, id))
        return tvb_to_spikeNet_interfaces
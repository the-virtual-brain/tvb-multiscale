# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from pandas import Series
import numpy as np

from tvb_multiscale.config import CONFIGURED, initialize_logger
from tvb_multiscale.interfaces.spikeNet_to_tvb_interface import SpikeNetToTVBinterface

from tvb.contrib.scripts.utils.data_structures_utils import property_to_fun


LOG = initialize_logger(__name__)


@add_metaclass(ABCMeta)
class SpikeNetToTVBInterfaceBuilder(object):
    _build_target_class = SpikeNetToTVBinterface
    interfaces = []
    spiking_nodes = Series()
    spiking_nodes_ids = []
    tvb_nodes_ids = []
    tvb_model = None
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

    @abstractmethod
    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        pass

    def build_interface(self, interface):
        # One SpikeNet output device for every combination of SpikeNet node
        # and TVB state variable/parameter to be transmitted
        # from SpikeNet to TVB
        spikeNet_to_tvb_interface = Series()
        spiking_nodes = interface.get("nodes", self.spiking_nodes_ids)  # Indices corresponding to NEST nodes
        if spiking_nodes is None:
            spiking_nodes = self.spiking_nodes_ids
        spiking_nodes = list(spiking_nodes)
        if self.exclusive_nodes:
            # TODO: decide about the following: can a TVB node be updated from a NEST node via a NEST -> TVB interface,
            # get simulated in TVB and again update SpikeNet via a TVB -> SpikeNet interface?
            # Will it depend on whether there is also a directly coupling of that NEST node with other NEST nodes?
            assert np.all(spiking_node not in self.tvb_nodes_ids for spiking_node in spiking_nodes)
        interface_weights = np.ones((len(spiking_nodes),)).astype("f")
        interface_weight_fun = property_to_fun(interface.pop("interface_weights", 1.0))
        delays = np.ones((len(spiking_nodes),)).astype("f")
        delay_fun = property_to_fun(interface.pop("delays", 0.0))
        for i_w, spiking_node in enumerate(spiking_nodes):
            interface_weights[i_w] = interface_weight_fun(spiking_node)
            delays[i_w] = delay_fun(spiking_node)
        # Delays should be set to the device
        interface["delays"] = delays  # Per node
        # Convert TVB node index to interface SpikeNet node index:
        spiking_nodes_ids = [np.where(self.spiking_nodes_ids == spiking_node)[0][0]
                             for spiking_node in spiking_nodes]
        interface["nodes"] = spiking_nodes_ids
        devices = self.build_and_connect_devices([interface], self.spiking_nodes)
        for name, device_set in devices.items():
            try:
                # The index of the TVB state variable that is targeted
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                tvb_sv_id = None  # DEPRECATED!: it might be a TVB parameter, not a state variable
                raise ValueError("tvb_sv_id=%s doesn't correspond "
                                 "to the index of a TVB state variable for interface %s!\n"
                                 % (str(interface.tvb_sv_id), str(name)))
            spikeNet_to_tvb_interface[name] = \
                self._build_target_class(self.spiking_network, tvb_sv_id, nodes_ids=spiking_nodes,
                                         scale=interface_weights).from_device_set(device_set, name)
        return spikeNet_to_tvb_interface

    def build_interfaces(self):
        spikeNet_to_tvb_interfaces = Series()
        for interface in self.interfaces:
            spikeNet_to_tvb_interfaces = \
                spikeNet_to_tvb_interfaces.append(self.build_interface(interface))
        return spikeNet_to_tvb_interfaces

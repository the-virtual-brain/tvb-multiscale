# -*- coding: utf-8 -*-

from six import string_types
import numpy as np
from tvb_nest.config import NESTconfig
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.simulator_nest.nest_factory import build_and_connect_output_devices, build_input_device
from tvb_nest.simulator_nest.models.network import NESTNetwork
from tvb_nest.simulator_nest.models.devices import NESTInputDeviceDict
from tvb_nest.interfaces.tvb_to_nest_interface import TVBtoNESTinterface
from tvb_nest.interfaces.tvb_to_nest_parameter_interface import TVBNESTParameterInterface, NEST_INPUT_PARAMETERS
from tvb_nest.interfaces.nest_to_tvb_interface import NESTtoTVBinterface
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict, OrderedDict

LOG = initialize_logger(__name__)


class TVBNESTInterfaceBuilder(object):
    config = NESTconfig()
    tvb_model = None
    integrator = None
    connectivity = None
    tvb_nodes_ids = []

    nest_nodes_ids = []
    nest_network = []
    # The NEST nodes the activity of which is transformed to TVB state variables
    nest_to_tvb_interfaces = []
    # # Example from RedWWexcIOinhIBuilder
    # # For transmitting NEST rates to TVB state variables:
    # connections = OrderedDict({})
    # connections["S_e"] = "E"
    # connections["S_i"] = "I"
    # nest_to_tvb_interfaces = [{"model": "spike_detector",
    #                           "properties": NEST_OUTPUT_DEVICES_PROPS_DEF["device_type"],
    #                           "connections": {"S_e": "E",
    #                                           "S_i": "I"}}]
    # # For transmitting NEST rates to TVB parameters:
    # connections = OrderedDict({})
    # connections["r_e"] = "E"
    # connections["r_i"] = "I"
    # nest_to_tvb_interfaces = [{"model": "spike_detector",
    #                           "properties": NEST_OUTPUT_DEVICES_PROPS_DEF["device_type"],
    #                           "connections": {"r_e": "E",
    #                                           "r_i": "I"}}]

    # The NEST nodes where TVB input is directed
    tvb_to_nest_interfaces = []

    # Example from RedWWexcIOinhIBuilder
    # # For directly setting an external current parameter in NEST neurons:
    # tvb_to_nest_interfaces = [{"model": "current", "parameter": "I_e", "sign": 1,
    #                            "connections": {"S_e": ["E", "I"]}}]
    # # # For injecting current to NEST neurons via a dc generator:
    # # tvb_to_nest_interfaces = [{"model": "dc_generator", "sign": 1,
    # #                            "connections": {"S_e": ["E", "I"]}}]

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids,
                 nest_to_tvb_interfaces=None, tvb_to_nest_interfaces=None, config=NESTconfig()):
        self.config = config
        if isinstance(nest_network, NESTNetwork):
            self.nest_network = nest_network
        else:
            raise ValueError("Input nest_network is not a NESTNetwork object!\n%s" % str(nest_network))
        if isinstance(tvb_simulator, Simulator):
            self.connectivity = tvb_simulator.connectivity
            self.integrator = tvb_simulator.integrator
            self.tvb_model = tvb_simulator.model
            self.tvb_nodes_ids = list(range(self.connectivity.weights.shape[0]))
            try:
                self.nest_nodes_ids = ensure_list(nest_nodes_ids)
                for i_n in self.nest_nodes_ids:
                    self.tvb_nodes_ids.remove(i_n)
            except:
                raise ValueError("Failed to compute tvb_nodes_ids from nest_nodes_ids %s "
                                 "and TVB connectivity of size %s!"
                                 % (str(self.nest_nodes_ids, self.connectivity.weights.shape[0])))
            if tvb_simulator.stimulus is not None:
                if np.sum(tvb_simulator.stimulus[:, self.nest_nodes_ids]):
                    raise ValueError("TVB-NEST interface does not implement TVB stimulus application to NEST nodes!\n"
                                     "The user has to configure such stimulus as part of the NEST model, "
                                     "via appropriate NEST input devices!")
        else:
            raise ValueError("Input simulator_tvb is not a Simulator object!\n%s" % str(tvb_simulator))

        if nest_to_tvb_interfaces is not None:
            self.nest_to_tvb_interfaces = ensure_list(nest_to_tvb_interfaces)
        if tvb_to_nest_interfaces is not None:
            self.tvb_to_nest_interfaces = ensure_list(tvb_to_nest_interfaces)

    @property
    def nest_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def nest_instance(self):
        return self.nest_network.nest_instance

    @property
    def nest_nodes(self):
        return self.nest_network.region_nodes

    def assert_delay(self, delay):
        return np.maximum(self.nest_min_delay, delay)

    def build_nest_to_tvb_interfaces(self):
        # One NEST output device for every combination of NEST mode
        # and TVB state variable/parameter to be transmitted
        # from NEST to TVB
        devices = build_and_connect_output_devices(self.nest_instance,
                                                   self.nest_to_tvb_interfaces,
                                                   self.nest_nodes, config=self.config)
        interfaces = IndexedOrderedDict(OrderedDict({}))
        for name, device_set in devices.items():
            try:
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                tvb_sv_id = None  # it might be a TVB parameter, not a state variable
            interfaces.update({name: NESTtoTVBinterface(name, device_set.model, OrderedDict({})). \
                              from_device_set(device_set, tvb_sv_id)})
        return interfaces

    def _build_tvb_to_nest_parameter_input(self, interface):
        # One interface for every combination NEST node
        # and TVB state variable to be transmitted
        # from TVB to NEST
        connections = interface["connections"]
        if isinstance(connections, string_types):
            connections = {connections: slice(None)}  # return all population types
        interfaces = IndexedOrderedDict(OrderedDict({}))
        default_parameter = NEST_INPUT_PARAMETERS[interface["model"]]
        for name, populations in connections.items():
            try:
                tvb_coupling_id = self.tvb_model.cvar.tolist().index(
                    self.tvb_model.state_variables.index(name))
            except:
                raise ValueError("Failed to compute the coupling index of TVB state variable %s!" % name)
            interfaces.update({name: TVBNESTParameterInterface(self.nest_instance,
                                                               name,
                                                               interface["model"],
                                                               interface.get("parameter", default_parameter),
                                                               OrderedDict({}),
                                                               tvb_coupling_id,
                                                               interface.get("sign", 1))})
            for node in self.nest_nodes.values():
                interfaces[name].update({node.label: node[populations]})
        return interfaces

    def _build_tvb_to_nest_input_devices(self, interface):
        # One NEST stimulation device for every combination of
        # TVB node and state variable to be transmitted from TVB to NEST
        model = interface["model"]
        connections = interface["connections"]
        sign = interface.get("sign", 1)
        if isinstance(connections, string_types):
            connections = {connections: slice(None)}  # return all population types
        interfaces = IndexedOrderedDict(OrderedDict({}))
        for name, populations in connections.items():
            try:
                tvb_sv_id = self.tvb_model.state_variables.index(name)
            except:
                raise ValueError("Failed to compute the index of TVB state variable %s!" % name)
            interfaces.update({name: TVBtoNESTinterface(name, model, tvb_sv_id=tvb_sv_id)})
            for tvb_id in self.tvb_nodes_ids:
                # Generate a device for every TVB node to be represented in NEST network...
                try:
                    input_device = build_input_device(self.nest_instance, interface, config=self.config)
                except:
                    raise ValueError("Failed to create NEST device %s!" % model)
                # ...and connect it to every NEST node
                for i_node, node in enumerate(self.nest_nodes.values()):
                    # ...with the corresponding weight, sign, and delay
                    weight = sign * self.connectivity.weights[self.nest_nodes_ids[i_node], tvb_id]
                    delay = self.assert_delay(
                        self.connectivity.delays[
                            self.nest_nodes_ids[i_node], tvb_id])
                    self.nest_instance.Connect(input_device.device, node[populations],
                                               syn_spec={"weight": weight, "delay": delay})
                input_device.update_number_of_connections()
                interfaces[name].update({self.connectivity.region_labels[tvb_id]: input_device})
        return interfaces

    def _build_tvb_to_nest_interfaces(self):
        # This method will create the necessary NEST input devices and connect them to their target NEST nodes.
        # If the nest proxy doesn't need to be a device, we just add it to the list
        tvb_to_nest_interfaces = IndexedOrderedDict({})
        # Create a list of input devices for every TVB node inside NEST and connect them to the target NEST nodes:
        for interface in self.tvb_to_nest_interfaces:
            model = interface.get("model", None)
            if model in NESTInputDeviceDict.keys():
                tvb_to_nest_interfaces.update(self._build_tvb_to_nest_input_devices(interface))
            else:
                tvb_to_nest_interfaces.update(self._build_tvb_to_nest_parameter_input(interface))
        return tvb_to_nest_interfaces

    def build_interface(self, tvb_nest_interface):
        """
        Configure the TVB NEST interface of the fine scale as well other aspects of its interface with TVB
        :return: tvb_nest_interface object
        """

        tvb_nest_interface.config = self.config
        tvb_nest_interface.nest_instance = self.nest_instance
        # TODO: find out why the model instance is different in simulator and interface...
        tvb_nest_interface.tvb_model = self.tvb_model
        tvb_nest_interface.dt = self.integrator.dt
        tvb_nest_interface.tvb_nodes_ids = self.tvb_nodes_ids
        tvb_nest_interface.nest_nodes_ids = self.nest_nodes_ids
        tvb_nest_interface.nest_network = self.nest_network

        tvb_nest_interface.nest_to_tvb_interfaces = self.build_nest_to_tvb_interfaces()
        tvb_nest_interface.tvb_to_nest_interfaces = self._build_tvb_to_nest_interfaces()

        return tvb_nest_interface

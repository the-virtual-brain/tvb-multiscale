# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_nest.interfaces.builders.tvb_to_nest_interface_device_builder import TVBtoNESTInterfaceDeviceBuilder
from tvb_nest.interfaces.builders.tvb_to_nest_interface_parameter_builder import TVBtoNESTInterfaceParameterBuilder
from tvb_nest.interfaces.builders.nest_to_tvb_interface_builder import NESTtoTVBInterfaceBuilder
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.simulator_nest.models.network import NESTNetwork
from tvb_nest.simulator_nest.models.devices import NESTInputDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list

LOG = initialize_logger(__name__)


class TVBNESTInterfaceBuilder(object):
    tvb_model = None
    integrator = None
    connectivity = None
    tvb_nodes_ids = []
    nest_nodes_ids = []
    exclusive_nodes = False
    nest_network = []

    # The NEST nodes where TVB input is directed
    tvb_to_nest_interfaces = []

    # Example from RedWWexcIOinhIBuilder
    # # For directly setting an external current parameter in NEST neurons instantaneously:
    # tvb_to_nest_interfaces = [{ "model": "current",  "parameter": "I_e",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
    #                            "interface_weights": 1.0,
    # ------------------------------------------------------------------------------------------------------------------
    # #                                       TVB sv -> NEST population
    #                            "connections": {"S_e": ["E", "I"]},
    #                            "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    # # tvb_to_nest_interfaces = [{"model": "dc_generator", "params": {},
    #                              "interface_weights": 1.0,  # Applied outside NEST for each interface device
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                              "weights": 1.0,  # To multiply TVB connectivity weight
    # #                            To add to TVB connectivity delay:
    #                              "delays": self.nodes_min_delay ,
    # #                                       TVB sv -> NEST population
    #                              "connections": {"S_e": ["E", "I"]},
    #                              "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    # tvb_to_nest_interfaces =  [{"model": "poisson_generator", "params": {},
    #                            "interface_weights": 1.0,  # Applied outside NEST for each interface device
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                            "weights": 1.0,  # To multiply TVB connectivity weight
    # #                          To add to TVB connectivity delay:
    #                            "delays": self.nodes_min_delay,
    #                            "receptor_types": 0,
    # ------------------------------------------------------------------------------------------------------------------
    # #                                       TVB sv -> NEST population
    #                            "connections": {"S_e": ["E", "I"]},
    #                            "source_nodes": None, "target_nodes": None}]  # None means all here

    # The NEST nodes the activity of which is transformed to TVB state variables or parameters
    nest_to_tvb_interfaces = []
    # # Example from RedWWexcIOinhIBuilder
    # # For transmitting NEST rates to TVB parameters instantaneously:
    # connections = OrderedDict({})
    # connections["r_e"] = "E"
    # connections["r_i"] = "I"
    # nest_to_tvb_interfaces = [{"model": "spike_detector",
    #                           "params": NEST_OUTPUT_DEVICES_PROPS_DEF["device_type"],
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
    #                           "interface_weights": 1.0,
    #                           "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
    #                           "connections": {"r_e": "E",
    #                                           "r_i": "I"}, "nodes": None}]  # None means all here

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):
        if isinstance(nest_network, NESTNetwork):
            self.nest_network = nest_network
        else:
            raise ValueError("Input nest_network is not a NESTNetwork object!\n%s" % str(nest_network))
        self.exclusive_nodes = exclusive_nodes
        if isinstance(tvb_simulator, Simulator):
            self.connectivity = tvb_simulator.connectivity
            self.integrator = tvb_simulator.integrator
            self.tvb_model = tvb_simulator.model
            self.nest_nodes_ids = np.array(ensure_list(nest_nodes_ids))
            self.tvb_nodes_ids = list(range(self.connectivity.weights.shape[0]))
            if self.exclusive_nodes:
                try:
                    for i_n in self.nest_nodes_ids:
                        self.tvb_nodes_ids.remove(i_n)
                except:
                    raise ValueError("Failed to compute tvb_nodes_ids from nest_nodes_ids %s "
                                     "and TVB connectivity of size %s!"
                                     % (str(self.nest_nodes_ids, self.connectivity.number_of_regions)))
            self.tvb_nodes_ids = np.array(self.tvb_nodes_ids)
            if tvb_simulator.stimulus is not None:
                if np.sum(tvb_simulator.stimulus[:, self.nest_nodes_ids.tolist()]):
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
    def nest_instance(self):
        return self.nest_network.nest_instance

    @property
    def config(self):
        return self.nest_network.config.nest

    @property
    def nest_nodes(self):
        return self.nest_network.region_nodes

    @property
    def nodes_min_delay(self):
        return self.nest_network.nodes_min_delay

    @property
    def nest_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def tvb_dt(self):
        return self.integrator.dt

    def assert_delay(self, delay):
        return np.maximum(self.nest_min_delay, delay)

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
        tvb_nest_interface.exclusive_nodes = self.exclusive_nodes
        tvb_nest_interface.nest_network = self.nest_network

        tvb_nest_interface.tvb_to_nest_interfaces = Series({})
        # Create a list of input devices for every TVB node inside NEST and connect them to the target NEST nodes:
        for interface in self.tvb_to_nest_interfaces:
            model = interface.get("model", None)
            if model in NESTInputDeviceDict.keys():
                tvb_nest_interface.tvb_to_nest_interfaces = \
                    tvb_nest_interface.tvb_to_nest_interfaces.append(
                        TVBtoNESTInterfaceDeviceBuilder([],
                                                        self.nest_instance, self.nest_nodes, self.nest_nodes_ids,
                                                        self.tvb_nodes_ids, self.tvb_model,
                                                        self.connectivity, self.tvb_dt).build_interface(interface)
                                                                    )
            else:
                tvb_nest_interface.tvb_to_nest_interfaces = \
                    tvb_nest_interface.tvb_to_nest_interfaces.append(
                            TVBtoNESTInterfaceParameterBuilder([],
                                                               self.nest_instance, self.nest_nodes, self.nest_nodes_ids,
                                                               self.tvb_nodes_ids, self.tvb_model).
                                                                                        build_interface(interface)
                                                                    )

        tvb_nest_interface.nest_to_tvb_interfaces = \
            NESTtoTVBInterfaceBuilder(self.nest_to_tvb_interfaces,
                                      self.nest_instance, self.nest_nodes, self.nest_nodes_ids,
                                      self.tvb_nodes_ids, self.tvb_model, self.connectivity).build()

        return tvb_nest_interface

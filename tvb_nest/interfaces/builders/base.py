# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_nest.interfaces.builders.tvb_to_nest_device_interface_builder import TVBtoNESTDeviceInterfaceBuilder
from tvb_nest.interfaces.builders.tvb_to_nest_parameter_interface_builder import TVBtoNESTParameterInterfaceBuilder
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

    # TVB <-> NEST transformations' weights/funs
    # If set as weights, they will become a transformation function of
    # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
    # If set as a function of lambda state: fun(state), it will become a vector function of:
    # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
    # TVB -> NEST
    w_tvb_to_spike_rate = 1000.0  # (spike rate in NEST is in spikes/sec, assuming TVB rate is spikes/ms)
    w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
    w_tvb_to_potential = 1.0  # assuming mV in both NEST and TVB
    # TVB <- NEST
    w_spikes_to_tvb_rate = 1.0  # (assuming spikes/ms in TVB and given the division of spikes' number with dt in ms)
    w_spikes_to_tvb = 1.0
    w_spikes_var_to_tvb = 1.0
    w_potential_to_tvb = 1.0

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
            self.tvb_simulator = tvb_simulator
            self.tvb_weights = self.tvb_connectivity.scaled_weights(mode='region')
            self.nest_nodes_ids = np.array(ensure_list(nest_nodes_ids))
            self.tvb_nodes_ids = list(range(self.tvb_connectivity.weights.shape[0]))
            if self.exclusive_nodes:
                try:
                    for i_n in self.nest_nodes_ids:
                        self.tvb_nodes_ids.remove(i_n)
                except:
                    raise ValueError("Failed to compute tvb_nodes_ids from nest_nodes_ids %s "
                                     "and TVB connectivity of size %s!"
                                     % (str(self.nest_nodes_ids), self.tvb_connectivity.number_of_regions))
            self.tvb_nodes_ids = np.array(self.tvb_nodes_ids)
            if self.tvb_simulator.stimulus is not None:
                if np.sum(self.tvb_simulator.stimulus[:, self.nest_nodes_ids.tolist()]):
                    raise ValueError("TVB-NEST interface does not implement TVB stimulus application to NEST nodes!\n"
                                     "The user has to configure such stimulus as part of the NEST model, "
                                     "via appropriate NEST input devices!")
        else:
            raise ValueError("Input simulator_tvb is not a Simulator object!\n%s" % str(tvb_simulator))

        # TVB <-> NEST transformations' weights/funs
        # If set as weights, they will become a transformation function of
        # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
        # If set as a function of lambda state: fun(state), it will become a vector function of:
        # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
        # TVB -> NEST
        self.w_tvb_to_spike_rate = 1000.0  # (spike rate in NEST is in spikes/sec, whereas dt is in ms)
        self.w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
        self.w_tvb_to_potential = 1.0  # assuming mV in both NEST and TVB
        # TVB <- NEST
        self.w_spikes_to_tvb_rate = 1.0  # (assuming spikes/ms in TVB)
        self.w_spikes_to_tvb = 1.0
        self.w_spikes_var_to_tvb = 1.0
        self.w_potential_to_tvb = 1.0

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
    def logger(self):
        return self.nest_network.logger

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
    def tvb_delays(self):
        return self.tvb_simulator.connectivity.delays

    @property
    def tvb_connectivity(self):
        return self.tvb_simulator.connectivity

    @property
    def tvb_integrator(self):
        return self.tvb_simulator.integrator

    @property
    def tvb_model(self):
        return self.tvb_simulator.model

    @property
    def tvb_dt(self):
        return self.tvb_simulator.integrator.dt

    @property
    def number_of_nodes(self):
        return self.tvb_connectivity.number_of_regions

    def assert_delay(self, delay):
        return np.maximum(self.nest_min_delay, delay)

    # TODO: find a possible way to differentiate scalings between
    #  receiver (as in _tvb_state_to_nest_current),
    #  and sender (as in all other cases below), node indexing

    def _prepare_tvb_to_nest_transform_fun(self, prop, dummy):
        transform_fun = prop.split("w_")[1]
        if hasattr(getattr(self, prop), "__call__"):
            # If the property is already set as a function:
            return {transform_fun:
                lambda state_variable, region_nodes_indices=None: \
                    getattr(self, prop)(state_variable[region_nodes_indices])}
        else:
            # If the property is set just as a weight:
            setattr(self, prop, dummy * getattr(self, prop))
            return {transform_fun:
                lambda state_variable, region_nodes_indices, weights=getattr(self, prop): \
                    state_variable[region_nodes_indices] * weights[region_nodes_indices]}

    def _prepare_nest_to_tvb_transform_fun(self, prop, dummy):
        transform_fun = prop.split("w_")[1]
        if hasattr(getattr(self, prop), "__call__"):
            # If the property is already set as a function:
            return {transform_fun:
                lambda nest_variable, region_nodes_indices=None:
                    getattr(self, prop)(nest_variable)}
        else:
            # If the property is set just as a weight:
            setattr(self, prop, dummy * getattr(self, prop))
            return {transform_fun:
                lambda nest_variable, region_nodes_indices, weights=getattr(self, prop): \
                    nest_variable * weights[region_nodes_indices]}

    def generate_transforms(self):
        self.w_spikes_to_tvb /= self.tvb_dt  # Used to convert number of spikes to a spike rate
        self.w_spikes_to_tvb_rate /= self.tvb_dt  # Used to convert number of spikes to a spike rate
        dummy = np.ones((self.number_of_nodes, ))
        # Confirm good shape for TVB-NEST interface model parameters
        # TODO: find a possible way to differentiate scalings between
        #  receiver (as in _tvb_state_to_nest_current),
        #  and sender (as in all other cases below), node indexing
        #  Also, the size doesn't have to be in all cases equal to number_of_nodes,
        #  but sometimes equal to number_of_nest_nodes or to number_of_tvb_nodes
        transforms = {}
        for prop in ["w_tvb_to_current",
                     "w_tvb_to_potential",
                     "w_tvb_to_spike_rate"]:
            transforms.update(self._prepare_tvb_to_nest_transform_fun(prop, dummy))
        for prop in ["w_spikes_to_tvb_rate",
                     "w_spikes_to_tvb",
                     "w_spikes_var_to_tvb",
                     "w_potential_to_tvb"]:
            transforms.update(self._prepare_nest_to_tvb_transform_fun(prop, dummy))
        return transforms

    def build_interface(self, tvb_nest_interface):
        """
        Configure the TVB NEST interface of the fine scale as well other aspects of its interface with TVB
        :return: tvb_nest_interface object
        """

        tvb_nest_interface.config = self.config
        tvb_nest_interface.nest_instance = self.nest_instance
        # TODO: find out why the model instance is different in simulator and interface...
        tvb_nest_interface.tvb_model = self.tvb_model
        tvb_nest_interface.dt = self.tvb_dt
        tvb_nest_interface.tvb_nodes_ids = self.tvb_nodes_ids
        tvb_nest_interface.nest_nodes_ids = self.nest_nodes_ids
        tvb_nest_interface.exclusive_nodes = self.exclusive_nodes
        tvb_nest_interface.nest_network = self.nest_network

        tvb_nest_interface.transforms = self.generate_transforms()

        tvb_nest_interface.tvb_to_nest_interfaces = Series({})
        # Create a list of input devices for every TVB node inside NEST and connect them to the target NEST nodes:
        for interface in self.tvb_to_nest_interfaces:
            model = interface.get("model", None)
            if model in NESTInputDeviceDict.keys():
                tvb_nest_interface.tvb_to_nest_interfaces = \
                    tvb_nest_interface.tvb_to_nest_interfaces.append(
                        TVBtoNESTDeviceInterfaceBuilder([],
                                                        self.nest_instance, self.nest_nodes, self.nest_nodes_ids,
                                                        self.tvb_nodes_ids, self.tvb_model,
                                                        self.tvb_weights, self.tvb_delays,
                                                        self.tvb_connectivity.region_labels, self.tvb_dt,
                                                        self.exclusive_nodes).build_interface(interface)
                                                                    )
            else:
                tvb_nest_interface.tvb_to_nest_interfaces = \
                    tvb_nest_interface.tvb_to_nest_interfaces.append(
                            TVBtoNESTParameterInterfaceBuilder([],
                                                               self.nest_instance, self.nest_nodes,
                                                               self.nest_nodes_ids, self.tvb_nodes_ids,
                                                               self.tvb_model, self.exclusive_nodes).
                                                                                        build_interface(interface)
                                                                    )

        tvb_nest_interface.nest_to_tvb_interfaces = \
            NESTtoTVBInterfaceBuilder(self.nest_to_tvb_interfaces,
                                      self.nest_instance, self.nest_nodes, self.nest_nodes_ids,
                                      self.tvb_nodes_ids, self.tvb_model, self.exclusive_nodes).build()

        return tvb_nest_interface

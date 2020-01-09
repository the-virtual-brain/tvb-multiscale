# -*- coding: utf-8 -*-
import numpy as np
from tvb_nest.nest_models.devices import NESTInputDeviceDict
from tvb_nest.interfaces.builders.tvb_to_nest_devices_interface_builder import TVBtoNESTDeviceInterfaceBuilder
from tvb_nest.interfaces.builders.tvb_to_nest_parameter_interface_builder import TVBtoNESTParameterInterfaceBuilder
from tvb_nest.interfaces.builders.nest_to_tvb_interface_builder import NESTtoTVBInterfaceBuilder
from tvb_multiscale.interfaces.builders.base import TVBSpikeNetInterfaceBuilder


class TVBNESTInterfaceBuilder(TVBSpikeNetInterfaceBuilder):
    _tvb_to_spikNet_device_interface_builder = TVBtoNESTDeviceInterfaceBuilder
    _tvb_to_spikeNet_parameter_interface_builder = TVBtoNESTParameterInterfaceBuilder
    _spikeNet_to_tvb_interface_builder = NESTtoTVBInterfaceBuilder
    _input_device_dict = NESTInputDeviceDict

    # TVB <-> Spiking Network transformations' weights/funs
    # If set as weights, they will become a transformation function of
    # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
    # If set as a function of lambda state: fun(state), it will become a vector function of:
    # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
    # TVB -> Spiking Network
    w_tvb_to_spike_rate = 1000.0  # (e.g., spike rate in NEST is in spikes/sec, assuming TVB rate is spikes/ms)
    w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
    w_tvb_to_potential = 1.0  # assuming mV in both Spiking Network and TVB
    # TVB <- Spiking Network
    # We return from a Spiking Network spike_detector the ratio number_of_population_spikes / number_of_population_neurons
    # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
    # as long as a neuron cannot fire twice during a TVB time step, i.e.,
    # as long as the TVB time step (usually 0.001 to 0.1 ms)
    # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
    # For conversion to a rate, one has to do:
    # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
    # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
    w_spikes_to_tvb = 1.0
    w_spikes_var_to_tvb = 1.0
    # We return from a Spiking Network multimeter or voltmeter the membrane potential in mV
    w_potential_to_tvb = 1.0

    # Example from RedWWexcIOinhIBuilder
    # # For directly setting an external current parameter in NEST neurons instantaneously:
    # tvb_to_spikeNet_interfaces = [{ "model": "current",  "parameter": "I_e",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
    #                            "interface_weights": 1.0,
    # ------------------------------------------------------------------------------------------------------------------
    # #                                       TVB sv -> NEST population
    #                            "connections": {"S_e": ["E", "I"]},
    #                            "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    # # tvb_to_spikeNet_interfaces = [{"model": "dc_generator", "params": {},
    #                              "interface_weights": 1.0,  # Applied outside NEST for each interface device
    # # --Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None, **kwargs)------
    #                              "weights": 1.0,  # To multiply TVB connectivity weight
    # #                            To add to TVB connectivity delay:
    #                              "delays": self.spikeNet_min_delay ,
    # #                                       TVB sv -> NEST population
    #                              "connections": {"S_e": ["E", "I"]},
    #                              "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    # tvb_to_spikeNet_interfaces =  [{"model": "poisson_generator", "params": {},
    #                            "interface_weights": 1.0,  # Applied outside NEST for each interface device
    # # --Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None, **kwargs)------
    #                            "weights": 1.0,  # To multiply TVB connectivity weight
    # #                          To add to TVB connectivity delay:
    #                            "delays": self.spikeNet_min_delay,
    #                            "receptor_types": 0,
    # ------------------------------------------------------------------------------------------------------------------
    # #                                       TVB sv -> NEST population
    #                            "connections": {"S_e": ["E", "I"]},
    #                            "source_nodes": None, "target_nodes": None}]  # None means all here

    # The Spiking Network nodes the activity of which is transformed to TVB state variables or parameters
    # # Example from RedWWexcIOinhIBuilder
    # # For transmitting Spiking Network rates to TVB parameters instantaneously:
    # connections = OrderedDict({})
    # connections["r_e"] = "E"
    # connections["r_i"] = "I"
    # spikeNet_to_tvb_interfaces = [{"model": "spike_detector",
    #                           "params": NEST_OUTPUT_DEVICES_PROPS_DEF["device_type"],
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
    #                           "interface_weights": 1.0,
    #                           "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
    #                           "connections": {"r_e": "E",
    #                                           "r_i": "I"}, "nodes": None}]  # None means all here

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    @property
    def config(self):
        return self.spiking_network.config

    @property
    def spikeNet_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def nest_min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    def assert_delay(self, delay):
        return np.maximum(self.spikeNet_min_delay, delay)

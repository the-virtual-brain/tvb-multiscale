# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models.generic_2d_oscillator import Generic2dOscillator
from tvb_nest.simulator_tvb.models.generic_2d_oscillator import Generic2dOscillator as TVBGeneric2dOscillator


class Generic2DOscillatorBuilder(TVBNESTInterfaceBuilder):
    tvb_model = TVBGeneric2dOscillator()

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):

        if tvb_to_nest_interfaces is None:
    # For directly setting membrane potential V_m in NEST neurons instantaneously:
            tvb_to_nest_interfaces = [{"model": "potential",  "parameter": "V_m",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
                                       "interface_weights": 1.0,
    # ----------------------------------------------------------------------------------------------------------------
    #                                               TVB sv -> NEST population
                                       "connections": {"V": ["E"]},
                                       "nodes": None}]  # None means all here

    # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["V_m"] = ["E"]
            nest_to_tvb_interfaces = \
                [{"model": "voltmeter", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        super(Generic2DOscillatorBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                         tvb_to_nest_interfaces, nest_to_tvb_interfaces)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, Generic2dOscillator):
            tvb_nest_interface = Generic2dOscillator()
        return super(Generic2DOscillatorBuilder, self).build_interface(tvb_nest_interface)

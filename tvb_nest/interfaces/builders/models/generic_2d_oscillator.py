# -*- coding: utf-8 -*-

import numpy as np
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import Generic2dOscillator
from tvb_multiscale.spiking_models.builders.templates import tvb_weight, tvb_delay, receptor_by_source_region
from tvb_multiscale.simulator_tvb.models.generic_2d_oscillator import Generic2dOscillator as TVBGeneric2dOscillator


class Generic2DOscillatorBuilder(TVBNESTInterfaceBuilder):
    tvb_model = TVBGeneric2dOscillator()

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None,
                 E_L=-70.0, V_reset=-60.0, V_th=-55.0, g=5.0, V_low=-5.0):
        super(Generic2DOscillatorBuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids,
                                                         exclusive_nodes,
                                                         tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        if tvb_to_nest_interfaces is None:
    # For directly setting membrane potential V_m in NEST neurons instantaneously:
            self.tvb_to_spikeNet_interfaces = [{
                                       "model": "current",  "parameter": "I_e",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
                                       "interface_weights": 1.0,
    # ----------------------------------------------------------------------------------------------------------------
    #                                               TVB sv -> NEST population
                                       "connections": {"V": ["E", "I"]},
                                       "nodes": None}]  # None means all here

    # TODO: Find out why this leads to instability of integration!
    # # The NEST nodes the activity of which is transformed to TVB state variables or parameters
    #     if spikeNet_to_tvb_interfaces is None:
    #         # NEST -> TVB:
    #         # Use S_e and S_i instead of r_e and r_i
    #         # for transmitting to the TVB state variables directly
    #         connections = OrderedDict()
    #         #            TVB <- NEST
    #         connections["V_m"] = ["E"]
    #         self.spikeNet_to_tvb_interfaces = \
    #             [{"model": "voltmeter", "params": {},
    # # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
    #               "weights": 1.0, "delays": 0.0,
    # # ------------------------------------------------------------------------------------------------------------------
    #               "connections": connections, "nodes": None}]  # None means all here

        self.E_L = - np.abs(E_L)
        self.V_reset = - np.abs(V_reset)
        self.V_th = - np.abs(V_th)
        self.g = np.abs(g)
        self.V_low = np.abs(V_low)
        self.w_tvb_to_nest = np.abs(self.V_reset - self.E_L)  # ~ -60 - (-70) = 10
        self.w_nest_to_tvb = np.abs(self.V_th - self.V_reset)  # ~ -55 - (-60) = 5
        # TODO: confirm the following:
        #                            g * weight * abs(E_L-V_th) * Vtvb->nest
        self.w_tvb_to_current = lambda coupling: self.g * self.w_tvb_to_nest * np.maximum(-self.V_low, coupling)
        self.w_potential_to_tvb = lambda V_m: (V_m - self.V_reset) / self.w_nest_to_tvb

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, Generic2dOscillator):
            tvb_nest_interface = Generic2dOscillator()
        return super(Generic2DOscillatorBuilder, self).build_interface(tvb_nest_interface)

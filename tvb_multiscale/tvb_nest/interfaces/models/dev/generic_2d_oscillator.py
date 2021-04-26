# -*- coding: utf-8 -*-

import numpy as np
from tvb_multiscale.tvb_nest.interfaces.models.default import DefaultInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models import Generic2dOscillator


class Generic2DOscillatorBuilder(DefaultInterfaceBuilder):

    _tvb_nest_interface = Generic2dOscillator

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None,
                 E_L=-70.0, V_reset=-60.0, V_th=-55.0, g=5.0, V_low=-5.0):
        super(Generic2DOscillatorBuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids,
                                                         exclusive_nodes,
                                                         tvb_to_nest_interfaces, nest_to_tvb_interfaces)
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

    def build_rate_tvb_to_nest_interfaces(self):
        raise NotImplementedError

    def build_dc_tvb_to_nest_interfaces(self):
        raise NotImplementedError

    def build_param_tvb_to_nest_interfaces(self):
        self._build_default_param_tvb_to_nest_interfaces({"V": ["E", "I"]})

    def build_default_nest_to_tvb_interfaces(self):
        # TODO: Find out why this leads to instability of integration!
        self._build_default_nest_to_tvb_interfaces({"V_m": ["E"]})

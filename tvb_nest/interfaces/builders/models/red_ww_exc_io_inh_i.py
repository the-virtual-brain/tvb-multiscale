# -*- coding: utf-8 -*-

from tvb_nest.interfaces.builders.models.default import DefaultMultiSynapseInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class RedWWexcIOinhIBuilder(DefaultMultiSynapseInterfaceBuilder):

    _tvb_nest_interface = RedWWexcIOinhI

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, N_E=160, N_I=40):
        super(RedWWexcIOinhIBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                    tvb_to_nest_interfaces, nest_to_tvb_interfaces, N_E, N_I)

        self.global_coupling_scaling *= self.tvb_simulator.model.G[0].item()
        self.lamda = self.tvb_model.lamda[0].item()

        self.w_tvb_to_current = 1000 * self.tvb_model.J_N[0]  # (nA of TVB -> pA of NEST)
        # WongWang model parameter r is in Hz, just like poisson_generator assumes in NEST:
        self.w_tvb_to_spike_rate = 1.0
        # We return from a NEST spike_detector the ratio number_of_population_spikes / number_of_population_neurons
        # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
        # as long as a neuron cannot fire twice during a TVB time step, i.e.,
        # as long as the TVB time step (usually 0.001 to 0.1 ms)
        # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
        # For conversion to a rate, one has to do:
        # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
        # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
        # given WongWang model parameter r is in Hz but tvb dt is in ms:
        self.w_spikes_to_tvb = 1000.0 / self.tvb_dt

    # No random jitter to weights and delays by default for this model

    def tvb_weight_E_fun(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights, scale=self.global_coupling_scaling)

    def tvb_weight_I_fun(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights, scale=self.lamda * self.global_coupling_scaling)

    def tvb_delay_fun(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_fun(self, source_node, target_node, start=1):
        return super(RedWWexcIOinhIBuilder, self).receptor_fun(source_node, target_node, start)

    def build_default_rate_tvb_to_nest_interfaces(self):
        # The rate interface requires uniform weight = 1.0,
        # because spikes synapses are weighted with  parameter w_E_ext within the neuron model
        self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["E"]}, weights=1.0)

        if self.lamda:
            self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["I"]}, weights=1.0)

    def build_default_current_tvb_to_nest_interfaces(self):
        # Instead, the current dc interface requires TVB weights
        self._build_default_current_tvb_to_nest_interfaces({"S_e": ["E"]}, weights=self.tvb_weight_E_fun)
        if self.lamda > 0.0:
            self._build_default_current_tvb_to_nest_interfaces({"S_e": ["I"]},
                                                               interface_weights=1.0 * self.N_E / self.N_I,
                                                               weights=self.tvb_weight_I_fun)

    def build_default_param_tvb_to_nest_interfaces(self):
        self._build_default_param_tvb_to_nest_interfaces({"S_e": ["E"]})
        if self.lamda > 0.0:
            self._build_default_param_tvb_to_nest_interfaces({"S_e": ["I"]},
                                                             interface_weights=self.lamda * self.N_E / self.N_I)

    def build_default_nest_to_tvb_interfaces(self):
        self._build_default_nest_to_tvb_interfaces({"R_e": ["E"], "R_i": ["I"]})

# -*- coding: utf-8 -*-

from tvb_nest.interfaces.builders.models.default import DefaultInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay, receptor_by_source_region


class RedWWexcIOinhIBuilder(DefaultInterfaceBuilder):

    _tvb_nest_interface = RedWWexcIOinhI

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, N_E=160, N_I=40):
        super(RedWWexcIOinhIBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                    tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        self.global_coupling_scaling *= self.tvb_simulator.model.G[0].item()
        self.N_E = N_E
        self.N_I = N_I
        #       The total external excitatory current (tvb_mean_field_current * N_E)
        #       will be distributed to the neurons of exc and inh pops if lamda>0 (feeedforward inhibition)
        self._exc_weight = 1.0 / (1.0 + self.tvb_model.lamda[0]) * self.N_E
        self._inh_weight = (1.0 - self._exc_weight) * self.N_E

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

    def build_default_rate_tvb_to_nest_interfaces(self):
        self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["E"]},
                                                        interface_weights=self._exc_weight)

        if self._inh_weight:
            self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["I"]},
                                                            interface_weights=self._inh_weight)

    def build_default_current_tvb_to_nest_interfaces(self):
        self._build_default_current_tvb_to_nest_interfaces({"S_e": ["E"]},
                                                           interface_weights=self._exc_weight/self.N_E)
        if self._inh_weight > 0.0:
            self._build_default_current_tvb_to_nest_interfaces({"S_e": ["I"]},
                                                               interface_weights=self._inh_weight/self.N_I)

    def build_default_param_tvb_to_nest_interfaces(self):
        self._build_default_param_tvb_to_nest_interfaces({"S_e": ["E"]},
                                                         interface_weights=self._exc_weight/self.N_E)
        if self._inh_weight > 0.0:
            self._build_default_param_tvb_to_nest_interfaces({"S_e": ["I"]},
                                                             interface_weights=self._inh_weight/self.N_I)

    def build_default_nest_to_tvb_interfaces(self):
        self._build_default_nest_to_tvb_interfaces({"R_e": ["E"], "R_i": ["I"]})


class RedWWexcIOinhIMultisynapseBuilder(RedWWexcIOinhIBuilder):

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, N_E=160, N_I=160):
        super(RedWWexcIOinhIMultisynapseBuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids,
                                                                exclusive_nodes,
                                                                tvb_to_nest_interfaces, nest_to_tvb_interfaces,
                                                                N_E, N_I)

    def receptor_fun(self, source_node, target_node, start=3):
        return receptor_by_source_region(source_node, target_node, start)


class WWDeco2014Builder(RedWWexcIOinhIMultisynapseBuilder):

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, N_E=160, N_I=40):
        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes,
                                                tvb_to_nest_interfaces, nest_to_tvb_interfaces, N_E, N_I)

    # No random jitter to weights and delays by default for this model

    def build_default_rate_tvb_to_nest_interfaces(self):
        self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["E"]}, weights=1.0,
                                                        interface_weights=self._exc_weight)

        if self._inh_weight:
            self._build_default_rate_tvb_to_nest_interfaces({"R_e": ["I"]}, weights=1.0,
                                                            interface_weights=self._inh_weight)

    def tvb_weight(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights, scale=self.global_coupling_scaling)

    def tvb_delay(self, source_node, target_node, sigma=0.1):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_fun(self, source_node, target_node, start=1):
        return receptor_by_source_region(source_node, target_node, start)

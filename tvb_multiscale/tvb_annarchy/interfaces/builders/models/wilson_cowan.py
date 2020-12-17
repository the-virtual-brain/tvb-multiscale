# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.interfaces.builders.models.default import DefaultInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models import WilsonCowan

from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class WilsonCowanBuilder(DefaultInterfaceBuilder):

    _tvb_annarchy_interface = WilsonCowan

    def __init__(self, tvb_simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, populations_sizes=[100, 100]):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes,
                                                 tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        self.N_E = populations_sizes[0]
        self.N_I = populations_sizes[1]

        # WilsonCowan model state variables are bounded in [0, 1],
        # and have to be converted in Hz as poisson_generator assumes in NEST:
        self.w_tvb_to_spike_rate = 1000.0
        # We return from a ANNarchy spike_monitor the ratio number_of_population_spikes / number_of_population_neurons
        # for every TVB time step, which is already a quantity in the range [0.0, 1.0],
        # as long as a neuron cannot fire twice during a TVB time step, i.e.,
        # as long as the TVB time step (usually 0.001 to 0.1 ms)
        # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
        self.w_spikes_to_tvb = 1.0

    def build_default_rate_tvb_to_annarchy_interfaces(self):
        self._build_default_rate_tvb_to_annarchy_interfaces({"E": ["E", "I"]})

    # def build_dc_tvb_to_nest_interfaces(self):
    #     raise NotImplementedError

    def build_param_tvb_to_annarchy_interfaces(self):
        raise NotImplementedError

    def build_default_annarchy_to_tvb_interfaces(self):
        self._build_default_annarchy_to_tvb_interfaces({"Ein": ["E"], "Iin": ["I"]})


class WilsonCowanMultisynapseBuilder(WilsonCowanBuilder):

    def __init__(self, tvb_simulator, nest_network, annarchy_nodes_ids, exclusive_nodes=False,
                 tvb_to_annarchy_interfaces=None, annarchy_to_tvb_interfaces=None, populations_sizes=[100, 100]):
        super(WilsonCowanMultisynapseBuilder, self).__init__(tvb_simulator, nest_network, annarchy_nodes_ids,
                                                             exclusive_nodes,
                                                             tvb_to_annarchy_interfaces, annarchy_to_tvb_interfaces,
                                                             populations_sizes=populations_sizes)

    def receptor_fun(self, source_node, target_node, start=3):
        return receptor_by_source_region(source_node, target_node, start)

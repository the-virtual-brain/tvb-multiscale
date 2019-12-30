# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import WilsonCowan
from tvb_multiscale.spiking_models.builders.templates import tvb_weight, tvb_delay, receptor_by_source_region
from tvb_multiscale.simulator_tvb.models.wilson_cowan_constraint import WilsonCowan as TVBWilsonCowan

class WilsonCowanMultisynapseBuilder(TVBNESTInterfaceBuilder):
    tvb_model = TVBWilsonCowan()

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):
        super(WilsonCowanMultisynapseBuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids,
                                                             exclusive_nodes,
                                                             tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        if tvb_to_nest_interfaces is None:

    #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    # Options:
    # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
    # For spike trains with correlation probability p_copy set:
    # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
    # An alternative option to poisson_generator is:
    # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
            self.tvb_to_spikeNet_interfaces = [{
                                       "model": "inhomogeneous_poisson_generator",
                                       "params": {"allow_offgrid_times": False},
    # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                                        "interface_weights": 1000.0,  # Applied outside NEST for each interface device
    # #                                 A function of TVB connectivity weight
                                        "weights": self.tvb_weight,  # To multiply TVB connectivity weight
    # #                                 A function of TVB connectivity delay
                                        "delays": self.tvb_delay,
                                        "receptor_types": self.receptor_by_source_region,
    # ----------------------------------------------------------------------------------------------------------------
    #                                        TVB sv or param -> NEST population
                                        "connections": {"E": ["E", "I"]},
                                        "source_nodes": None, "target_nodes": None}]  # None means all here

    # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["E"] = ["E"]
            connections["I"] = ["I"]
            self.spikeNet_to_tvb_interfaces = \
                [{"model": "spike_detector", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        # WilsonCowan model state variables are bounded in [0, 1],
        # and have to be converted in Hz as poisson_generator assumes in NEST:
        self.w_tvb_to_spike_rate = 1000.0
        # We return from a NEST spike_detector the ratio number_of_population_spikes / number_of_population_neurons
        # for every TVB time step, which is already a quantity in the range [0.0, 1.0],
        # as long as a neuron cannot fire twice during a TVB time step, i.e.,
        # as long as the TVB time step (usually 0.001 to 0.1 ms)
        # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
        self.w_spikes_to_tvb = 1.0

    def tvb_weight(self, source_node, target_node):
        return tvb_weight(source_node, target_node, self.tvb_weights)

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_by_source_region(self, source_node, target_node, start=3):
        return receptor_by_source_region(source_node, target_node, start)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, WilsonCowan):
            tvb_nest_interface = WilsonCowan()
        return super(WilsonCowanMultisynapseBuilder, self).build_interface(tvb_nest_interface)

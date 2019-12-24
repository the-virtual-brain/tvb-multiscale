# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import WilsonCowan
from tvb_nest.simulator_tvb.models.wilson_cowan_constraint import WilsonCowan as TVBWilsonCowan


class WilsonCowanMultisynapseBuilder(TVBNESTInterfaceBuilder):
    tvb_model = TVBWilsonCowan()

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):

        if tvb_to_nest_interfaces is None:

    #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    # Options:
    # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
    # For spike trains with correlation probability p_copy set:
    # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
    # An alternative option to poisson_generator is:
    # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
            tvb_to_nest_interfaces = [{"model": "inhomogeneous_poisson_generator",
                                       "params": {"allow_offgrid_times": False},
    # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                                        "interface_weights": 1.0,  # Applied outside NEST for each interface device
                                        "weights": 1000.0,  # To multiply TVB connectivity weight
    #                                 To add to TVB connectivity delay:
                                        "delays": nest_network.nodes_min_delay,
                                        "receptor_types": lambda tvb_node_id, nest_node_id: int(tvb_node_id + 3),
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
            nest_to_tvb_interfaces = \
                [{"model": "spike_detector", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        super(WilsonCowanMultisynapseBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                 tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        # WilsonCowan model state variables are bounded in [0, 1],
        # and have to be converted in Hz as poisson_generator assumes in NEST:
        self.w_tvb_to_spike_rate = 1000.0
        # We return from a NEST spike_detector the ratio number_of_population_spikes / number_of_population_neurons
        # for every TVB time step, which is already a quantity in the range [0.0, 1.0],
        # as long as a neuron cannot fire twice during a TVB time step, i.e.,
        # as long as the TVB time step (usually 0.001 to 0.1 ms)
        # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
        self.w_spikes_to_tvb = 1.0

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, WilsonCowan):
            tvb_nest_interface = WilsonCowan()
        return super(WilsonCowanMultisynapseBuilder, self).build_interface(tvb_nest_interface)

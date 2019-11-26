# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import WilsonCowan
from tvb_nest.simulator_tvb.models.wilson_cowan_constraint import WilsonCowan as TVBWilsonCowan


class WilsonCowanBuilder(TVBNESTInterfaceBuilder):
    tvb_model = TVBWilsonCowan()

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):

        if tvb_to_nest_interfaces is None:

    #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
            tvb_to_nest_interfaces = [{"model": "poisson_generator", "params": {},
    # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                                        "interface_weights": 1.0,  # Applied outside NEST for each interface device
                                        "weights": 0.1,  # To multiply TVB connectivity weight
    #                                 To add to TVB connectivity delay:
                                        "delays": nest_network.nodes_min_delay,
                                        "receptor_types": 0,
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

        super(WilsonCowanBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                 tvb_to_nest_interfaces, nest_to_tvb_interfaces)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, WilsonCowan):
            tvb_nest_interface = WilsonCowan()
        return super(WilsonCowanBuilder, self).build_interface(tvb_nest_interface)

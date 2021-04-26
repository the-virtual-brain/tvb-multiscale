# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_multiscale.core.config import CONFIGURED
from tvb_multiscale.tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models import RedWWexcIOinhI


class RateWWAMPANMDAGABABuilder(TVBNESTInterfaceBuilder):

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, config=CONFIGURED):

        if tvb_to_nest_interfaces is None:
    # # For directly setting an external current parameter in NEST neurons instantaneously:
            tvb_to_nest_interfaces = [{"model": "current", "parameter": "I_e",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
                                       "interface_weights": 1.0,
    # ------------------------------------------------------------------------------------------------------------------
    #                                                   TVB  ->  NEST
                                       "connections": {"S_e": ["AMPA", "NMDA", "GABA"]},
                                       "nodes": None}]  # None means all here
    # TVB -> NEST:
    #
    # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    #       tvb_to_spikeNet_interfaces = [{"model": "dc_generator",
    #                                  "interface_weights": 1.0, # Applied outside NEST for each interface device
    #                                    "weights": 1.0,  # To multiply TVB connectivity weight
    # #                                To add to TVB connectivity delay:
    #                                  "delays": spiking_network.spikeNet_min_delay,            #
    #                                                 TVB  ->  NEST
    #                                 "connections": {"S_e": ["AMPA", "NMDA", "GABA"]},
    #                                 "source_nodes": None, "target_nodes": None}]  # None means all here



        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["R_e"] = ["AMPA", "NMDA"]
            connections["R_i"] = "GABA"
            nest_to_tvb_interfaces = [{"model": "spike_multimeter", "params": {},
                                       "connections": connections, "nodes": None}]  # None means all here

        super(RateWWAMPANMDAGABABuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes,
                                                        tvb_to_nest_interfaces, nest_to_tvb_interfaces, config)
        self.w_tvb_to_current = 1000 * self.tvb_model.J_N[0]  # (nA of TVB -> pA of NEST)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, RedWWexcIOinhI):
            tvb_nest_interface = RedWWexcIOinhI()
        return super(RateWWAMPANMDAGABABuilder, self).build_interface(tvb_nest_interface)

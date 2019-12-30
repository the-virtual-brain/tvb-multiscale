# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_multiscale.config import CONFIGURED
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


class RedRateWWexcIOinhIBuilder(TVBNESTInterfaceBuilder):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, config=CONFIGURED):
        if tvb_to_nest_interfaces is None:
        # For directly setting an external current parameter in NEST neurons instantaneously:
                tvb_to_nest_interfaces = [{"model": "current",  "parameter": "I_e",
        # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
                                           "interface_weights": 1.0,
        # ----------------------------------------------------------------------------------------------------------------
        #                                               TVB sv -> NEST population
                                           "connections": {"S_e": ["E", "I"]},
                                           "nodes": None}]  # None means all here

        # # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
        #         tvb_to_spikeNet_interfaces = [{"model": "dc_generator", "params": {},
        # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
        #                                    "interface_weight": 1.0,  # Applied outside NEST for each interface device
        #                                    "weights": 1.0,  # To multiply TVB connectivity weight
        # #                                 To add to TVB connectivity delay:
        # #                                   "delays": spiking_network.nodes_min_delay,
        # # ----------------------------------------------------------------------------------------------------------------
        # #                                                 TVB sv -> NEST population
        #                                    "connections": {"S_e": ["E", "I"]},
        #                                    "source_nodes": None, "target_nodes": None}]  # None means all here

        # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["S_e"] = "E"
            connections["S_i"] = "I"
            params = self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]
            params["record_from"] = ["S"]
            # TODO: decide about this!
            params["interval"] = tvb_simulator.integrator.dt
            nest_to_tvb_interfaces = \
                [{"model": "multimeter",
        # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
        # ------------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        super(RedRateWWexcIOinhIBuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes,
                                                        tvb_to_nest_interfaces, nest_to_tvb_interfaces, config)
        self.w_tvb_to_current = 1000 * self.tvb_model.J_N[0]  # (nA of TVB -> pA of NEST)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, RedWWexcIOinhI):
            tvb_nest_interface = RedWWexcIOinhI()
        return super(RedRateWWexcIOinhIBuilder, self).build_interface(tvb_nest_interface)

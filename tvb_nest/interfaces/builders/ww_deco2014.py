# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.config import CONFIGURED
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models.red_ww_exc_io_inh_i import RedWWexcIOinhI
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


class WWDeco2014Builder(TVBNESTInterfaceBuilder):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, config=CONFIGURED):

        if tvb_to_nest_interfaces is None:
            # TVB -> NEST:
            #
            # 1.1. For current transmission from TVB to NEST,
            # either choose a NEST dc_generator device:
            # tvb_to_nest_interfaces = \
            #    [{"model": "dc_generator", "sign": 1,
            # #                      TVB  ->  NEST
            #      "connections": {"S_e": ["E", "I"]}}]

            # 1.2. or modify directly the external current stimulus parameter:
            tvb_to_nest_interfaces = \
                [{"model": "current", "parameter": "I_e", "sign": 1,
                  #                TVB  ->  NEST
                  "connections": {"S_e": ["E", "I"]}}]

            # 2.1. For spike transmission from TVB to NEST:
            # tvb_to_nest_interfaces = \
            #    [{"model": "poisson_generator", "sign": 1,
            # #                      TVB  ->  NEST
            #      "connections": {"S_e": ["E", "I"]}}]

        if nest_to_tvb_interfaces is None:
            connections = OrderedDict({})
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["r_e"] = ["E"]
            connections["r_i"] = ["I"]
            nest_to_tvb_interfaces = [{"model": "spike_detector",
                                       "params": config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
                                       "connections": connections}]

        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids,
                                                tvb_to_nest_interfaces, nest_to_tvb_interfaces, config)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, RedWWexcIOinhI):
            tvb_nest_interface = RedWWexcIOinhI()
        return super(WWDeco2014Builder, self).build_interface(tvb_nest_interface)

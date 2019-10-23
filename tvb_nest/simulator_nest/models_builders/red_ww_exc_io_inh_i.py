# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models_builders.base import NESTModelBuilder


class RedWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED):
        super(RedWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Spiking populations labels:
        self.populations_names = ["E", "I"]
        # Spiking populations scalings for the number of neurons:
        self.populations_scales = [1.0, 0.7]
        # Some properties for the default synapse to be used:
        self.default_connection["params"]["rule"] = "fixed_indegree"

        # Connection weights between the distinct populations:
        # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
        w_ee = 150.0
        w_ei = -1000.0
        w_ie = 150.0
        w_ii = -1000.0

        # Within region-node connections' weights
        self.population_connectivity_synapses_weights = \
            np.array([[w_ee, w_ei],     # exc_i -> exc_i, inh_i -> exc_i
                      [w_ie, w_ii]]).T  # exc_i -> inh_i, inh_i -> inh_i
        self.population_connectivity_synapses_delays = \
            np.array(self.tvb_dt / 4)

        # Among/Between region-node connections
        # Given that w_ee == w_ie = J_N,
        # and that only the excitatory population of one region-node couples to
        # both excitatory and inhibitory populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"src_population": "E", "trg_population": ["E", "I"],
              "model": self.default_connection["model"],
              "params": self.default_connection["params"],
              "weight": w_ee,  # weight scaling the TVB connectivity weight
              "delay": 0.0,
              "receptor_type": 0}]  # additional delay to the one of TVB connectivity

        # Creating spike_detector devices to be able to observe NEST spiking activity:
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        self.output_devices = \
            [{"model": "spike_detector",
              "props": config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
              "nodes": None, "connections": connections}]

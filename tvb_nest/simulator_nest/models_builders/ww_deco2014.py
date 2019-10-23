# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models_builders.base import NESTModelBuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules


class WWDeco2014Builder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, J_i=1.0):
        # Some properties for the default synapse to be used:
        self.default_connection["params"]["rule"] = "fixed_indegree"
        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        # Spiking populations labels:
        self.populations_names = ["E", "I"]
        self.populations_models = ["iaf_cond_deco2014"]
        nest_models = self.nest_instance.Models()
        for model in self.populations_models:
            if model not in nest_models:
                # If the model is not install into NEST already
                module = model + "module"
                try:
                    # Try to install it...
                    self.nest_instance.Install(module)
                except:
                    # ...unless we need to first compile it:
                    compile_modules(module, recompile=False, config=self.config)
                    # and now install it...
                    self.nest_instance.Install(module)

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Spiking populations scalings for the number of neurons:
        rcptr = self.nest_instance.GetDefaults('iaf_cond_deco2014')['receptor_types']
        self.populations_scales = [1.0, 0.7]

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.J_i = J_i
        w_ee = 1.4
        w_ei = -self.J_i
        w_ie = 1.0
        w_ii = -1.0

        # Within region-node connections' weights
        self.population_connectivity_synapses_weights = \
            np.array([[w_ee, w_ei],   # E -> E, I -> E
                      [w_ie, w_ii]])  # E -> I, I -> I
        self.population_connectivity_synapses_delays = self.tvb_dt / 4
        self.population_connectivity_synapses_receptor_types = \
            np.array([[(rcptr["SPIKESEXC_AMPA_REC"], rcptr["SPIKESEXC_NMDA"]), rcptr["SPIKESINH_GABA"]],
                      [(rcptr["SPIKESEXC_AMPA_REC"], rcptr["SPIKESEXC_NMDA"]), rcptr["SPIKESINH_GABA"]]]).T

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"src_population": "E", "trg_population": ["E", "I"],
              "model": self.default_connection["model"],
              "params": self.default_connection["params"],
              "weight": 1.0,  # weight scaling the TVB connectivity weight
              "delay": 0.0,  # additional delay to the one of TVB connectivity
              "receptor_type": rcptr["SPIKESEXC_AMPA_EXT"]},
             ]

        # Creating spike_detector devices to be able to observe NEST spiking activity:
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        props_multimeter = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]
        props_multimeter['record_from'] = ["V_m",
                                           "s_AMPA_ext", "s_AMPA_rec", "x_NMDA", "s_NMDA", "s_GABA",
                                           "I_AMPA_ext", "I_AMPA_rec", "I_NMDA", "I_GABA", "I_leak"]
        props_spike_detector = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"]
        self.output_devices = \
            [{"model": "multimeter", "props": props_multimeter, "nodes": None, "connections": connections},
             {"model": "spike_detector", "props": props_spike_detector, "nodes": None, "connections": connections}]

# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models_builders.base import NESTModelBuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules


class RedRateWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 J_N=150, J_i=1000):
        self.default_connection["model"] = "rate_connection"
        super(RedRateWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Spiking populations labels:
        self.populations_names = ["E", "I"]
        self.populations_models = ["tvb_rate_redwongwang_exc",
                                   "tvb_rate_redwongwang_inh"]
        nest_models = self.nest_instance.Models()
        for model in self.populations_models:
            if model not in nest_models:
                # If the model is not install into NEST already
                try:
                    # Try to install it...
                    self.nest_instance.Install("tvb_rate_wongwangmodule")
                except:
                    # ...unless we need to first compile it:
                    compile_modules("tvb_rate_wongwang", recompile=False, config=self.config)
                    # and now install it...
                    self.nest_instance.Install("tvb_rate_wongwangmodule")
                nest_models = self.nest_instance.Models()
        # Spiking populations scalings for the number of neurons:
        self.populations_scales = [1.0, 0.7]
        # Some properties for the default synapse to be used:
        self.default_connection["params"]["rule"] = "fixed_indegree"

        # Connection weights between the distinct populations:
        # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
        self.J_N = J_N
        self.J_i = J_i
        w_ee = J_N
        w_ei = - J_i
        w_ie = J_N
        w_ii = - J_i

        # Within region-node connections' weights
        self.population_connectivity_synapses_weights = \
            np.array([[w_ee, w_ei],   # exc_i -> exc_i, inh_i -> exc_i
                      [w_ie, w_ii]])  # exc_i -> inh_i, inh_i -> inh_i
        self.population_connectivity_synapses_delays = np.array(self.tvb_dt / 4)

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

        # Creating  devices to be able to observe NEST activity:
        connections = OrderedDict({})
        #          label <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params_multimeter = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]
        params_multimeter['record_from'] = ["S", "rate", "I_syn", "currents"]
        self.output_devices = \
            [{"model": "multimeter", "params": params_multimeter,
              "nodes": None, "connections": connections}]

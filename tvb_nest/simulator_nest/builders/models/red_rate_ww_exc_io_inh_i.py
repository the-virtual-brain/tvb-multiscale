# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.builders.base import NESTModelBuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules


class RedRateWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 J_N=150, J_i=1000):
        config.default_connection["model"] = "rate_connection"
        super(RedRateWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.J_N = J_N
        self.J_i = J_i

        # Compile if models are missing
        nest_models = self.nest_instance.Models()
        for model in ["tvb_rate_redwongwang_exc", "tvb_rate_redwongwang_inh"]:
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

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Populations' configurations
        self.populations = [{"label": "E", "model": "tvb_rate_redwongwang_exc",
                             "scale": 1, "params": {}, "nodes": None},  # None means "all"
                            {"label": "I", "model": "tvb_rate_redwongwang_inh",
                             "scale": 0.7, "params": {}, "nodes": None}]  # None means "all"

        # Within region-node connections
        # Connection weights between the distinct populations:
        # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
        self.populations_connections = [
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "params": self.default_populations_connection["params"],
             "weight": self.J_N, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "E", "target": "I",  # E -> I
             "model": self.default_populations_connection["model"],
             "params": self.default_populations_connection["params"],
             "weight": self.J_N, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "E",  # I -> E
             "model": self.default_populations_connection["model"],
             "params": self.default_populations_connection["params"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": self.default_populations_connection["model"],
             "params": self.default_populations_connection["params"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
        ]
        # Among/Between region-node connections
        # Given that w_ee == w_ie = J_N,
        # and that only the excitatory population of one region-node couples to
        # both excitatory and inhibitory populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"src_population": "E", "trg_population": ["E", "I"],
              "model": self.default_nodes_connection["model"],
              "params": self.default_nodes_connection["params"],
              "weight": self.J_N,  # weight scaling the TVB connectivity weight
              "delay": self.default_nodes_connection["delay"],  # additional delay to the one of TVB connectivity
              "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
             ]

        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        connections = OrderedDict({})
        #          label <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        params['record_from'] = ["S", "rate", "I_syn", "currents"]
        self.output_devices = [{"model": "multimeter", "params": params,
                                "connections": connections, "nodes": None}]  # None means "all"

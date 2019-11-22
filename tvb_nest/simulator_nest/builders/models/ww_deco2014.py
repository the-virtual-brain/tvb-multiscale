# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.builders.base import NESTModelBuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules


def params_E(node_index, nest_nodes_ids, tvb_weights, w_ee, J_i):
    return {"w_E_ext": tvb_weights[nest_nodes_ids.index(node_index)], "w_E": w_ee, "w_I": J_i}


class WWDeco2014Builder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 w_ee=1.4, J_i=1.0):
        config.nest.default_model = "iaf_cond_deco2014"
        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.w_ee = w_ee
        self.J_i = J_i

        # Compile if model is missing
        nest_models = self.nest_instance.Models()
        for model in ["iaf_cond_deco2014"]:
            if model not in nest_models:
                # If the model is not installed into NEST already
                module = model + "module"
                try:
                    # Try to install it...
                    self.nest_instance.Install(module)
                except:
                    # ...unless we need to first compile it:
                    compile_modules(model, recompile=False, config=self.config.nest)
                    # and now install it...
                    self.nest_instance.Install(module)
                nest_models = self.nest_instance.Models()

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Populations' configurations
        self.populations = [{"label": "E", "model": self.default_population["model"],
                             "params": lambda node_index: params_E(node_index, list(self.nest_nodes_ids),
                                                                   self.tvb_weights, self.J_i),
                             "scale": 1,  "nodes": None},  # None means "all"
                            {"label": "I", "model": self.default_population["model"],
                             "params": {"w_E": 1.0, "w_I": 1.0}, "scale": 0.7, "nodes": None}  # None means "all"
                            ]


        # Within region-node connections
        self.populations_connections = [
            {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "params": self.default_populations_connection["params"],
             "weight": self.w_ee,  "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "E", "target": "I",  # E -> I
              "model": self.default_populations_connection["model"],
              "params": self.default_populations_connection["params"],
              "weight": 1.0, "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "E",  # I -> E
              "model": self.default_populations_connection["model"],
              "params": self.default_populations_connection["params"],
              "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
              "model": self.default_populations_connection["model"],
              "params": self.default_populations_connection["params"],
              "weight": -1.0, "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None}  # None means "all"
                                    ]
        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.nodes_connections = [
            {"src_population": "E", "trg_population": ["E", "I"],
              "model": self.default_nodes_connection["model"],
              "params": self.default_nodes_connection["params"],
              "weight": 1.0,  # weight scaling the TVB connectivity weight
              "delay": self.default_nodes_connection["delay"]["delay"],  # additional delay to the one of TVB connectivity
              # Each region emits spikes in its own port:
              "receptor_type": lambda source_region_index, target_region_index=None: source_region_index+1,
              "source_nodes": None, "target_nodes": None}  # None means "all"
                                 ]

        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = []
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"])
        self.output_devices.append({"model": "spike_detector", "params": params,
                                    "connections": connections, "nodes": None})  # None means "all"
        connections = OrderedDict({})
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        params['record_from'] = ["V_m",
                                 "s_AMPA_ext", "s_AMPA_rec", "x_NMDA", "s_NMDA", "s_GABA",
                                 "I_AMPA_ext", "I_AMPA_rec", "I_NMDA", "I_GABA", "I_L"]
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means "all"


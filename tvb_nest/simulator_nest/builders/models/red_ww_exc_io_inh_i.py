# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.builders.base import NESTModelBuilder


class RedWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 w_ee=1.4, J_i=1.0):
        # config.DEFAULT_MODEL = "iaf_cond_deco2014"
        super(RedWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.w_ee = w_ee
        self.J_i = J_i

        # Common order of neurons' number per population:
        self.populations_order = 100
        # Populations' configurations
        self.populations = [{"label": "E", "model": self.default_population["model"],
                             "params": {}, "scale": 1,  "nodes": None},  # None means "all"
                            {"label": "I", "model": self.default_population["model"],
                             "params": {}, "scale": 0.7, "nodes": None}  # None means "all"
                            ]

        # Within region-node connections
        self.populations_connections = [
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
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
              "receptor_type": 0, "nodes": None},  # None means "all"
             ]

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
              "model": self.default_nodes_connection["model"],
              "params": self.default_nodes_connection["params"],
              "weight": 1.0,  # weight scaling the TVB connectivity weight
              "delay": self.default_nodes_connection["delay"],  # additional delay to the one of TVB connectivity
              # Each region emits spikes in its own port:
              "receptor_type": 0,  "source_nodes": None, "target_nodes": None}  # None means "all"
                                 ]

        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = []
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        self.output_devices.append({"model": "spike_detector", "params": {},
                                    "connections": connections, "nodes": None})  # None means all here
        connections = OrderedDict({})
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means all here

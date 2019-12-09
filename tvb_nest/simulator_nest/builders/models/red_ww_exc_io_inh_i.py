# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
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
        pop_weight_fun = lambda weight: {"distribution": "normal", "mu": weight, "sigma": 0.1*np.abs(weight)}
        self.populations_connections = [
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": pop_weight_fun(self.w_ee),
             "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "E", "target": "I",  # E -> I
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": pop_weight_fun(1.0),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "E",  # I -> E
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": pop_weight_fun(-self.J_i),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": pop_weight_fun(-1.0),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             ]

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        nodes_weight_fun = lambda source_nest_node_id=None, target_nest_node_id=None: \
            np.maximum(1.0, self.tvb_simulator.model.G[0] * (1.0 + 0.1 * np.random.normal()))
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": nodes_weight_fun(),  # weight scaling the TVB connectivity weight
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

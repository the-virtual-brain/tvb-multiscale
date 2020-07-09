# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import tvb_delay, scale_tvb_weight


class BasalGangliaIzhikevichBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED):
        super(BasalGangliaIzhikevichBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        self.default_population["model"] = "izhikevich"

        # Common order of neurons' number per population:
        self.population_order = 200

        self.populations = [
            {"label": "I", "model": self.default_population["model"],  # Igpe in [0, 1], Igpi in [2, 3]
             "params": {}, "nodes": [0, 1, 2, 3],  # None means "all"
             "scale": 1.0},
            {"label": "E", "model": self.default_population["model"],  # Estn in [4, 5], Eth in [8, 9]
             "params": {}, "nodes": [4, 5, 8, 9],  # None means "all"
             "scale": 1.0},
            {"label": "I1", "model": self.default_population["model"],  # Isd1 in [6, 7]
             "params": {}, "nodes": [6, 7],  # None means "all"
             "scale": 1.0},
            {"label": "I2", "model": self.default_population["model"],  # Isd2 in [6, 7]
             "params": {}, "nodes": [6, 7],  # None means "all"
             "scale": 1.0}
        ]

        synapse_model = self.default_populations_connection["model"]  # "static_synapse"
        # default connectivity spec:
        # conn_spec = {"autapses": True, 'multapses': True, 'rule': "all_to_all",
        #              "indegree": None, "outdegree": None, "N": None, "p": 0.1}
        conn_spec = self.default_populations_connection["conn_spec"]

        # Intra-regions'-nodes' connections
        self.populations_connections = []
        for pop in self.populations:
            # Only self-connections and only for all inhibitory  populations
            if pop["label"][0] == "I":
                self.populations_connections.append(
                    {"source": pop["label"], "target": pop["label"],
                     "model": synapse_model, "conn_spec": conn_spec,
                     "weight": -1.0, "delay": self.default_min_delay,  # 0.001
                     "receptor_type": 0, "nodes": pop["nodes"]})

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()
        # if we use Reduced Wong Wang model, we also need to multiply with the global coupling constant G:
        self.global_coupling_scaling *= self.tvb_simulator.model.G[0].item()

        # Inter-regions'-nodes' connections
        self.nodes_connections = []
        for src_pop, trg_pop, src_nodes, trg_nodes in \
            zip(
               # "Isd1->Igpi", "Isd2->Igpe", "Igpe->Igpi", "Igpi->Eth", "Igpe->Estn", "Eth->[Isd1, Isd2]", "Estn->[Igpe, Igpi]",
                ["I1",         "I2",         "I",          "I",         "I",          "E",                 "E"],  # source
                ["I",          "I",          "I",          "E",         "E",          ["I1", "I2"],        "I"],  # target
                [[6, 7],       [6, 7],       [0, 1],       [2, 3],      [0, 1],       [8, 9],              [4, 5]],  # source nodes
                [[2, 3],       [0, 1],       [2, 3],       [8, 9],      [4, 5],       [6, 7],              [0, 1, 2, 3]]):  # target nodes
            if src_pop[0] == "I":
                sign = -1.0
            else:
                sign = 1.0
            self.nodes_connections.append(
                {"source": src_pop, "target": trg_pop,
                 "model": self.default_nodes_connection["model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": lambda source_node, target_node: sign * self.tvb_weight_fun(source_node, target_node),
                 "delay": lambda source_node, target_node: self.tvb_delay_fun(source_node, target_node),
                 "receptor_type": 0, "source_nodes": src_nodes, "target_nodes": trg_nodes})

        # Creating  devices to be able to observe NEST activity:
        self.output_devices = []
        #          label <- target population
        for pop in self.populations:
            connections = OrderedDict({})
            connections[pop["label"] + "_spikes"] = pop["label"]
            self.output_devices.append(
                {"model": "spike_detector", "params": {},
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

        # Labels have to be different
        #               label    <- target population
        params = {"withtime": True, "withgid": True, 'record_from': ["V_m"], "interval": 1.0}
        for pop in self.populations:
            connections = OrderedDict({})
            connections[pop["label"] + '_V_m'] = pop["label"]
            self.output_devices.append(
                {"model": "multimeter", "params": params,
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to all

        # Create a spike stimulus input device
        self.input_devices = [
            {"model": "ac_generator",
             "params": {"frequency": 30.0, "phase": 0.0, "amplitude": 1.0, "offset": 0.0,
                        "start": 1.0},  # "stop": 100.0  "origin": 0.0,
             "connections": {"Stimulus": ["E"]},  # "Estn"
             "nodes": [4, 5],  # None means apply to all
             "weights": 1.0, "delays": 0.0}
        ]  #

    def tvb_weight_fun(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights, scale=self.global_coupling_scaling)

    def tvb_delay_fun(self, source_node, target_node):
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

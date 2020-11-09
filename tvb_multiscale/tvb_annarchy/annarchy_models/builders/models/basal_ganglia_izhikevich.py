# -*- coding: utf-8 -*-

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy.models import Izhikevich_Hamker
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyModelBuilder
from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight


class TVBWeightFun(object):
    tvb_weights = np.array([])
    global_coupling_scaling = 1.0

    def __init__(self, tvb_weights, global_coupling_scaling=1.0):
        self.tvb_weights = tvb_weights
        self.global_coupling_scaling = global_coupling_scaling

    def __call__(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.global_coupling_scaling)


class BasalGangliaIzhikevichBuilder(ANNarchyModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, annarchy_instance=None, config=CONFIGURED):
        super(BasalGangliaIzhikevichBuilder, self).__init__(tvb_simulator, nest_nodes_ids, annarchy_instance, config)
        self.default_population["model"] = Izhikevich_Hamker

        # Common order of neurons' number per population:
        self.population_order = 200

        self.params_common = {"E_ampa": 0.0, "E_gaba": -90.0, "v_th": 30.0, "c": -72.0,
                              "C": 1.0, "I": 0.0,
                              "tau_refrac": 10.0, "tau_syn": 1.0, "tau_ampa": 10.0, "tau_gaba": 10.0,
                              "n0": 140.0, "n1": 5.0, "n2": 0.04}
        self._paramsI = deepcopy(self.params_common)
        self._paramsI.update({"a": 0.005, "b": 0.585, "d": 4.0})
        self._paramsE = deepcopy(self.params_common)
        self.paramsStr = deepcopy(self.params_common)
        self.paramsStr.update({"v_th": 40.0, "C": 50.0,
                               "n0": 61.65119, "n1": 2.594639, "n2": 0.022799,
                               "a": 0.05, "b": -20.0, "c": -55.0, "d": 377.0})

        self.Igpe_nodes_ids = [0, 1]
        self.Igpi_nodes_ids = [2, 3]
        self.Estn_nodes_ids = [4, 5]
        self.Eth_nodes_ids = [8, 9]
        self.Istr_nodes_ids = [6, 7]

        self.Estn_stim = {"rate": 500.0, "weight": 0.009}
        self.Igpe_stim = {"rate": 100.0, "weight": 0.015}
        self.Igpi_stim = {"rate": 700.0, "weight": 0.02}

        self.populations = [
            {"label": "E", "model": self.default_population["model"],  # Estn in [4, 5], Eth in [8, 9]
             "params": self.paramsE, "nodes": self.Estn_nodes_ids + self.Eth_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "I", "model": self.default_population["model"],  # Igpe in [0, 1], Igpi in [2, 3]
             "params": self.paramsI, "nodes": self.Igpe_nodes_ids + self.Igpi_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "I1", "model": Izhikevich_Hamker,  # Isd1 in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "I2", "model": Izhikevich_Hamker,  # Isd2 in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0}
        ]
        populations_sizes = OrderedDict()
        for pop in self.populations:
            populations_sizes[pop["label"]] = int(np.round(pop["scale"] * self.population_order))

        synapse_model = self.default_populations_connection["synapse_model"]  # "DefaultSpikingSynapse"
        # default connectivity spec:
        # conn_spec= {'method': "all_to_all"}
        conn_spec = self.default_populations_connection["conn_spec"]
        conn_spec["allow_self_connections"] = True
        conn_spec["force_multiple_weights"] = False

        # Intra-regions'-nodes' connections
        self.populations_connections = []
        for pop in self.populations:
            # Only self-connections and only for all inhibitory  populations
            if pop["label"][0] == "I":
                self.populations_connections.append(
                    {"source": pop["label"], "target": pop["label"],
                     "synapse_model": synapse_model, "conn_spec": conn_spec,
                     "weight": 1.0, "delay": self.default_min_delay,  # 0.001
                     "receptor_type": "gaba", "nodes": pop["nodes"]})

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = tvb_simulator.coupling.a[0].item()
        # if we use Reduced Wong Wang model, we also need to multiply with the global coupling constant G:
        self.global_coupling_scaling *= tvb_simulator.model.G[0].item()

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
                target = "gaba"
            else:
                target = "ampa"
            self.nodes_connections.append(
                    {"source": src_pop, "target": trg_pop,
                     "synapse_model": self.default_nodes_connection["synapse_model"],
                     "conn_spec": self.default_nodes_connection["conn_spec"],
                     "weight": TVBWeightFun(self.tvb_weights, self.global_coupling_scaling),
                     "delay": lambda source_node, target_node: self.tvb_delay_fun(source_node, target_node),
                     "receptor_type": target, "source_nodes": src_nodes, "target_nodes": trg_nodes})

        # Creating  devices to be able to observe ANNarchy activity:
        self.output_devices = []
        params = self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["SpikeMonitor"]
        for pop in self.populations:
            connections = OrderedDict({})
            #                      label <- target population
            params["label"] = pop["label"] + "_spikes"
            connections[params["label"]] = pop["label"]
            self.output_devices.append(
                {"model": "SpikeMonitor", "params": deepcopy(params),
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

        # Labels have to be different for every connection to every distinct population
        # params for baladron implementation commented out for the moment
        # TODO: use baladron neurons
        params = self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["Monitor"]
        params.update({"period": 1.0,  'record_from': ["v", "u", "syn", "g_ampa", "g_gaba"]})
        for pop in self.populations:
            connections = OrderedDict({})
            #               label    <- target population
            connections[pop["label"]] = pop["label"]
            params["label"] = pop["label"]
            self.output_devices.append(
                {"model": "Monitor", "params": deepcopy(params),
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to all

        # Create a spike stimulus input device
        self.input_devices = [
            {"model": "PoissonPopulation",
             "params": {"rates": self.Estn_stim["rate"], "geometry": populations_sizes["E"], "name": "BaselineEstn"},
             "connections": {"BaselineEstn": ["E"]},  # "Estn"
             "nodes": self.Estn_nodes_ids,  # None means apply to all
             "weights": self.Estn_stim["weight"], "delays": 0.0, "receptor_type": "ampa"},
            {"model": "PoissonPopulation",
             "params": {"rates": self.Igpe_stim["rate"], "geometry": populations_sizes["I"], "name": "BaselineIgpe"},
             "connections": {"BaselineIgpe": ["I"]},  # "Igpe"
             "nodes": self.Igpe_nodes_ids,  # None means apply to all
             "weights": self.Igpe_stim["weight"], "delays": 0.0, "receptor_type": "ampa"},
            {"model": "PoissonPopulation",
             "params": {"rates": self.Igpi_stim["rate"], "geometry": populations_sizes["I"], "name": "BaselineIgpi"},
             "connections": {"BaselineIgpi": ["I"]},  # "Igpi"
             "nodes": self.Igpi_nodes_ids,  # None means apply to all
             "weights": self.Igpi_stim["weight"], "delays": 0.0, "receptor_type": "ampa"},
            # {"model": "ACCurrentInjector",
            #  "params": {"frequency": 30.0, "phase": 0.0, "amplitude": 1.0, "offset": 0.0},
            #  "connections": {"DBS_Estn": ["E"]},  # "Estn"
            #  "nodes": self.Estn_nodes_ids,  # None means apply to all
            #  "weights": 1.0, "delays": 0.0}
        ]

    def paramsI(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsI)
        if node_id in self.Igpe_nodes_ids:
            params.update({"I": 12.0})
        elif node_id in self.Igpi_nodes_ids:
            params.update({"I": 30.0})
        return params

    def paramsE(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsE)
        if node_id in self.Estn_nodes_ids:
            params.update({"a": 0.005, "b": 0.265, "d": 2.0, "I": 3.0})
        elif node_id in self.Eth_nodes_ids:
            params.update({"a": 0.02, "b": 0.25, "d": 0.05, "I": 3.5})
        return params

    def tvb_delay_fun(self, source_node, target_node):
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

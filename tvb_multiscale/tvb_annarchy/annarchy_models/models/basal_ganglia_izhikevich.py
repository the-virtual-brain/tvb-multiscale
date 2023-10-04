# -*- coding: utf-8 -*-

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy.models import Izhikevich_Hamker
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyNetworkBuilder


class TVBWeightFun(object):

    tvb_weights = np.array(list())
    global_coupling_scaling = 1.0

    def __init__(self, tvb_weights, global_coupling_scaling=1.0):
        self.tvb_weights = tvb_weights
        self.global_coupling_scaling = global_coupling_scaling

    def __call__(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.global_coupling_scaling)


class BasalGangliaIzhikevichBuilder(ANNarchyNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(BasalGangliaIzhikevichBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                            spiking_simulator, config, logger)
        self.default_population["model"] = Izhikevich_Hamker

        # Common order of neurons' number per population:
        self.population_order = 200

        self.params_common = {"E_ampa": 0.0, "E_gaba": -90.0, "v_th": 30.0, "c": -65.0,
                              "C": 1.0, "I": 0.0,
                              "tau_refrac": 10.0, "tau_syn": 1.0, "tau_ampa": 10.0, "tau_gaba": 10.0,
                              "n0": 140.0, "n1": 5.0, "n2": 0.04,
                              "v": -72.0, "u": -14.0,
                              "noise": 0.0}
        self._paramsI = deepcopy(self.params_common)
        self._paramsI.update({"a": 0.005, "b": 0.585, "d": 4.0,
                              "v": -70.0, "u": -18.55})
        self._paramsE = deepcopy(self.params_common)
        self._paramsE.update({"v": -70.0, "u": -18.55})
        self.paramsStr = deepcopy(self.params_common)
        self.paramsStr.update({"v_th": 40.0, "C": 50.0,
                               "n0": 61.65119, "n1": 2.594639, "n2": 0.022799,
                               "a": 0.05, "b": -20.0, "c": -55.0, "d": 377.0,
                               "v": -70.0, "u": -18.55})

        self.Igpe_nodes_ids = [0, 1]
        self.Igpi_nodes_ids = [2, 3]
        self.I_nodes = self.Igpe_nodes_ids + self.Igpi_nodes_ids
        self.Estn_nodes_ids = [4, 5]
        self.Eth_nodes_ids = [8, 9]
        self.E_nodes = self.Estn_nodes_ids + self.Eth_nodes_ids
        self.Istr_nodes_ids = [6, 7]

        self.scaleBGoptTOtvb = 0.00205875
        self.global_coupling_scaling = self.scaleBGoptTOtvb

        # Create a spike stimulus input device
        # When TVB is connected, we don't need any baseline stimulus
        self.Estn_stim = {"rate": 500.0, "weight": 0.009}
        self.Igpe_stim = {"rate": 100.0, "weight": 0.015}
        self.Igpi_stim = {"rate": 700.0, "weight": 0.02}

    def paramsE(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsE)
        if node_id in self.Estn_nodes_ids:
            params.update({"a": 0.005, "b": 0.265, "d": 2.0, "I": 3.0})
        elif node_id in self.Eth_nodes_ids:
            params.update({"a": 0.02, "b": 0.25, "d": 0.05, "I": 3.5})
        return params

    def paramsI(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsI)
        if node_id in self.Igpe_nodes_ids:
            params.update({"I": 12.0})
        elif node_id in self.Igpi_nodes_ids:
            params.update({"I": 30.0})
        return params

    def tvb_delay_fun(self, source_node, target_node):
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

    def set_populations(self):
        self.populations = [
            {"label": "E", "model": self.default_population["model"],  # Estn in [4, 5], Eth in [8, 9]
             "params": self.paramsE, "nodes": self.E_nodes,  # None means "all"
             "scale": 1.0},
            {"label": "I", "model": self.default_population["model"],  # Igpe in [0, 1], Igpi in [2, 3]
             "params": self.paramsI, "nodes": self.Igpe_nodes_ids + self.Igpi_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "IdSN", "model": Izhikevich_Hamker,  # Isd1 in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "IiSN", "model": Izhikevich_Hamker,  # Isd2 in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0}
        ]
        self.population_sizes = OrderedDict()
        for pop in self.populations:
            self.population_sizes[pop["label"]] = int(np.round(pop["scale"] * self.population_order))

    def set_populations_connections(self):
        # Intra-regions'-nodes' connections
        self.populations_connections = []
        for pop in self.populations:
            # Only self-connections and only for all inhibitory  populations
            if pop["label"][0] == "I":
                self.populations_connections.append(
                    {"source": pop["label"], "target": pop["label"],
                     "synapse_model": self.default_populations_connection["synapse_model"],
                     "conn_spec": self.default_populations_connection["conn_spec"],
                     "weight": self.scaleBGoptTOtvb, "delay": self.default_min_delay,
                     "receptor_type": "gaba", "nodes": pop["nodes"]})

    def set_nodes_connections(self):
        # Inter-regions'-nodes' connections
        self.nodes_connections = []
        for src_pop, trg_pop, src_nodes, trg_nodes in \
                zip(
                    # "IdSN->Igpi",        "IiSN->Igpe",
                    # "Igpe->Igpi",        "Igpi->Eth",
                    # "Eth->[IdSN, IiSN]",
                    # "Igpe->Estn",        "Estn->[Igpe, Igpi]",
                    ["IdSN", "IiSN",
                     "I", "I",
                     "E",
                     "I", "E"],  # source
                    ["I", "I",
                     "I", "E",
                     ["IdSN", "IiSN"],
                     "E", "I"],  # target
                    [self.Istr_nodes_ids, self.Istr_nodes_ids,
                     self.Igpe_nodes_ids, self.Igpi_nodes_ids,
                     self.Eth_nodes_ids,
                     self.Igpe_nodes_ids, self.Estn_nodes_ids],  # source nodes
                    [self.Igpi_nodes_ids, self.Igpe_nodes_ids,
                     self.Igpi_nodes_ids, self.Eth_nodes_ids,
                     self.Istr_nodes_ids,
                     self.Estn_nodes_ids, self.I_nodes]):  # target nodes
            self.nodes_connections.append(
                {"source": src_pop, "target": trg_pop,
                 "synapse_model": self.default_nodes_connection["synapse_model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": TVBWeightFun(self.tvb_weights, self.global_coupling_scaling),
                 "delay": lambda source_node, target_node: self.tvb_delay_fun(source_node, target_node),
                 "receptor_type": "gaba" if src_pop[0] == "I" else "ampa",
                 "source_nodes": src_nodes, "target_nodes": trg_nodes})

    def set_SpikeMonitors(self):
        params = deepcopy(self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["SpikeMonitor"])
        params["period"] = 1.0
        for pop in self.populations:
            connections = OrderedDict({})
            #                      label <- target population
            params["label"] = pop["label"]
            connections[params["label"]] = pop["label"]
            self.output_devices.append(
                {"model": "SpikeMonitor", "params": deepcopy(params),
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

    def set_Monitors(self):
        # Labels have to be different for every connection to every distinct population
        # params for baladron implementation commented out for the moment
        # TODO: use baladron neurons
        params = deepcopy(self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["Monitor"])
        # TODO: Find out why it fails for the rest of monitor variables:
        # "u", "I", "I_syn", "I_syn_ex", "I_syn_in",
        params.update({"period": 1.0, 'variables': ["v",  "g_exc"]})
        for pop in self.populations:
            connections = OrderedDict({})
            #               label    <- target population
            params["label"] = pop["label"] + "_ts"
            connections[params["label"]] = pop["label"]
            self.output_devices.append(
                {"model": "Monitor", "params": deepcopy(params),
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to all

    def set_output_devices(self):
        # Creating  devices to be able to observe ANNarchy activity:
        self.output_devices = []
        self.set_SpikeMonitors()
        self.set_Monitors()

    def set_input_devices(self):
        self.input_devices = [
            {"model": "PoissonPopulation",
             "params": {"rates": self.Estn_stim["rate"],
                        "geometry": self.population_sizes["E"], "name": "BaselineEstn"},
             "connections": {"BaselineEstn": ["E"]},  # "Estn"
             "nodes": self.Estn_nodes_ids,  # None means apply to all
             "weights": self.Estn_stim["weight"], "delays": self.default_min_delay, "receptor_type": "base"},
            {"model": "PoissonPopulation",
             "params": {"rates": self.Igpe_stim["rate"],
                        "geometry": self.population_sizes["I"], "name": "BaselineIgpe"},
             "connections": {"BaselineIgpe": ["I"]},  # "Igpe"
             "nodes": self.Igpe_nodes_ids,  # None means apply to all
             "weights": self.Igpe_stim["weight"], "delays": self.default_min_delay, "receptor_type": "base"},
            {"model": "PoissonPopulation",
             "params": {"rates": self.Igpi_stim["rate"],
                        "geometry": self.population_sizes["I"], "name": "BaselineIgpi"},
             "connections": {"BaselineIgpi": ["I"]},  # "Igpi"
             "nodes": self.Igpi_nodes_ids,  # None means apply to all
             "weights": self.Igpi_stim["weight"], "delays": self.default_min_delay, "receptor_type": "base"}
        ]

    def set_defaults(self):
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        self.set_input_devices()

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(BasalGangliaIzhikevichBuilder, self).build()

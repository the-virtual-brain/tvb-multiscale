# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from copy import deepcopy

import scipy.io as sio

import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight


class WeightFun(object):

    def __init__(self, wGPeGPe, wGPiGPi, GPe_nodes, GPi_nodes):
        self.wGPeGPe = - np.abs(wGPeGPe)
        self.wGPiGPi = - np.abs(wGPiGPi)
        self.GPe_nodes = GPe_nodes
        self.GPi_nodes = GPi_nodes

    def __call__(self, node):
        if node in self.GPe_nodes:
            return self.wGPeGPe  # GPe -> GPe
        elif node in self.GPi_nodes:
            return self.wGPiGPi  # GPi -> GPi
        else:
            raise ValueError("Inhibitory population that is neither in the GPe %s, nor in the GPi %s region nodes, "
                             "but in %s!" % (str(self.GPe_nodes), str(self.GPi_nodes), str(node)))


class TVBWeightFun(object):
    tvb_weights = np.array([])
    global_coupling_scaling = 1.0
    sign = 1

    def __init__(self, tvb_weights, global_coupling_scaling=1.0, sign=1):
        self.tvb_weights = tvb_weights
        self.global_coupling_scaling = global_coupling_scaling
        self.sign = sign

    def __call__(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.sign*self.global_coupling_scaling)


class BasalGangliaIzhikevichBuilder(NESTModelBuilder):

    output_devices_record_to = "ascii"

    path_to_conn_file = ""

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, path_to_conn_file=""):
        super(BasalGangliaIzhikevichBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        self.default_population["model"] = "izhikevich_hamker"
        self.path_to_conn_file = path_to_conn_file

        # Common order of neurons' number per population:
        self.population_order = 200

        self.params_common = {"E_rev_AMPA": 0.0, "E_rev_GABA_A": -90.0, "V_th": 30.0, "c": -65.0,
                              "C_m": 1.0, "I_e": 0.0,
                              "t_ref": 10.0, "tau_rise": 1.0, "tau_rise_AMPA": 10.0, "tau_rise_GABA_A": 10.0,
                              "n0": 140.0, "n1": 5.0, "n2": 0.04}
        self._paramsI = deepcopy(self.params_common)
        self._paramsI.update({"a": 0.005, "b": 0.585, "d": 4.0})
        self._paramsE = deepcopy(self.params_common)
        self.paramsStr = deepcopy(self.params_common)
        self.paramsStr.update({"V_th": 40.0, "C_m": 50.0,
                               "n0": 61.65119, "n1": 2.594639, "n2": 0.022799,
                               "a": 0.05, "b": -20.0, "c": -55.0, "d": 377.0})

        self.Igpe_nodes_ids = [0, 1]
        self.Igpi_nodes_ids = [2, 3]
        self.I_nodes = self.Igpe_nodes_ids + self.Igpi_nodes_ids
        self.Estn_nodes_ids = [4, 5]
        self.Eth_nodes_ids = [8, 9]
        self.E_nodes = self.Estn_nodes_ids + self.Eth_nodes_ids
        self.Istr_nodes_ids = [6, 7]

        self.Estn_stim = {"rate": 500.0, "weight": 0.009}
        self.Igpe_stim = {"rate": 100.0, "weight": 0.015}
        self.Igpi_stim = {"rate": 700.0, "weight": 0.02}

        self.populations = [
            {"label": "E", "model": self.default_population["model"],  # Estn in [4, 5], Eth in [8, 9]
             "params": self.paramsE, "nodes": self.E_nodes,  # None means "all"
             "scale": 1.0},
            {"label": "I", "model": self.default_population["model"],  # Igpe in [0, 1], Igpi in [2, 3]
             "params": self.paramsI, "nodes": self.I_nodes,  # None means "all"
             "scale": 1.0},
            {"label": "IdSN", "model": self.default_population["model"],  # IdSN in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0},
            {"label": "IiSN", "model": self.default_population["model"],  # IiSN in [6, 7]
             "params": self.paramsStr, "nodes": self.Istr_nodes_ids,  # None means "all"
             "scale": 1.0}
        ]


        if os.path.exists(self.path_to_conn_file):
            weights = sio.loadmat(self.path_to_conn_file)
            # weights start from index 20:
            # % loadedParams ={
            # %     	'D1GPi_probs': probs[0],
            # %     	'D1GPi_weights'  : weights[0],
            # %     	'D2GPe_probs'   : probs[1],
            # %     	'D2GPe_weights'  : weights[1],
            # %     	'GPeSTN_probs'   : probs[2],
            # %     	'GPeSTN_weights'  : weights[2],
            # %     	'STNGPe_probs'   : probs[3],
            # %     	'STNGPe_weights'  : weights[3],
            # %     	'STNGPi_probs'   : probs[4],
            # %     	'STNGPi_weights' : weights[4],
            # %     	'GPeGPi_probs'   : probs[5],
            # %     	'GPeGPi_weights'  : weights[5],
            # %     	'GPeGPe_probs'   : probs[6],
            # %     	'GPeGPe_weights'  : weights[6],
            # %     	'GPiGPi_probs'   : probs[7],
            # %     	'GPiGPi_weights'  : weights[7],
            # %     	'GPiThal_probs'   : probs[8],
            # %     	'GPiThal_weights'  : weights[8],
            # %     	'ThaliSN_probs'   : probs[9],
            # %     	'ThaliSN_weights'  : weights[9],
            # %     	'ThaldSN_probs'   : probs[10],
            # %     	'ThaldSN_weights'  : weights[10],
            # %     	'dSNdSN_probs'   : probs[11],
            # %     	'dSNdSN_weights'  : weights[11],
            # %     	'iSNiSN_probs'   : probs[12],
            # %     	'iSNiSN_weights'  : weights[12],
            # %     	'CdSN_probs'   : probs[13],
            # %     	'CdSN_weights'  : weights[13],
            # %     	'CiSN_probs'   : probs[14],
            # %     	'CiSN_weights'  : weights[14],
            # %     	'CSTN_probs'   : probs[15],
            # %     	'CSTN_weights'  : weights[15],
            # %     	'V1Inh_probs'    : probs[16],
            # %     	'V1Inh_weights'  : weights[16],
            # %     	'InhV1_probs'    : probs[17],
            # %     	'InhV1_weights'  : weights[17],
            # %     	'InhInh_probs'   : probs[18],
            # %     	'InhInh_weights'  : weights[18]}
            self.wGPeGPe = weights["X"][0, 6 + 19]   # "GPe" -> "GPe"
            self.wGPiGPi = weights["X"][0, 7 + 19]   # "GPi" -> "GPi"
            self.wdSNdSN = weights["X"][0, 11 + 19]  # "IdSN" -> "IdSN"
            self.wiSNiSN = weights["X"][0, 12 + 19]  # "IiSN" -> "IiSN"
            self.wThiSN = weights["X"][0, 9 + 19]    # "Eth" -> "IiSN"

        else:
            # Take the mean values of Maith et al paper:
            self.wGPeGPe = 2.34e-3  # "GPe" -> "GPe"
            self.wGPiGPi = 3.78e-3  # "GPi" -> "GPi"
            self.wdSNdSN = 3.56e-3  # "IdSN" -> "IdSN"
            self.wiSNiSN = 3.02e-3  # "IiSN" -> "IiSN"
            self.wThiSN = 1.30e-3  # "Eth" -> "IiSN"

        synapse_model = self.default_populations_connection["synapse_model"]  # "static_synapse"
        # default connectivity spec:
        # conn_spec = {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
        #              "indegree": None, "outdegree": None, "N": None, "p": 0.1}
        conn_spec = self.default_populations_connection["conn_spec"]

        # Intra-regions'-nodes' connections
        self.populations_connections = []
        for pop in self.populations:
            # Only self-connections and only for all inhibitory  populations
            if pop["label"][0] == "I":
                if pop["label"] == "IdSN":
                    weight = -np.abs(self.wdSNdSN)
                elif pop["label"] == "IiSN":
                    weight = -np.abs(self.wiSNiSN)
                else:
                    weight = WeightFun(self.wGPeGPe, self.wGPiGPi, self.Igpe_nodes_ids, self.Igpi_nodes_ids)
                self.populations_connections.append(
                    {"source": pop["label"], "target": pop["label"],
                     "synapse_model": synapse_model, "conn_spec": conn_spec,
                     "weight": weight, "delay": self.default_min_delay,
                     "receptor_type": 0, "nodes": pop["nodes"]})

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()
        # if we use Reduced Wong Wang model, we also need to multiply with the global coupling constant G:
        self.global_coupling_scaling *= self.tvb_simulator.model.G[0].item()

        # Inter-regions'-nodes' connections
        self.nodes_connections = []
        for src_pop, trg_pop, src_nodes, trg_nodes in \
            zip(
               # "IdSN->Igpi",        "IiSN->Igpe",
               # "Igpe->Igpi",        "Igpi->Eth",
               # "Eth->IdSN",         "Eth->IiSN",
               # "Igpe->Estn",        "Estn->[Igpe, Igpi]",
                ["IdSN",              "IiSN",
                 "I",                 "I",
                 "E",                 "E",
                 "I",                 "E"],                  # source
                ["I",                 "I",
                 "I",                 "E",
                 "IdSN",              "IiSN",
                 "E",                 "I"],                  # target
                [self.Istr_nodes_ids, self.Istr_nodes_ids,
                 self.Igpe_nodes_ids, self.Igpi_nodes_ids,
                 self.Eth_nodes_ids,  self.Eth_nodes_ids,
                 self.Igpe_nodes_ids, self.Estn_nodes_ids],  # source nodes
                [self.Igpi_nodes_ids, self.Igpe_nodes_ids,
                 self.Igpi_nodes_ids, self.Eth_nodes_ids,
                 self.Istr_nodes_ids, self.Istr_nodes_ids,
                 self.Estn_nodes_ids, self.I_nodes]):        # target nodes
            if src_pop[0] == "I":
                sign = -1
            else:
                sign = 1
            if src_pop == "E" and trg_pop == "IiSN":
                weight = sign * np.abs(self.wThiSN)
            else:
                weight = TVBWeightFun(self.tvb_weights, self.global_coupling_scaling, sign)
            self.nodes_connections.append(
                    {"source": src_pop, "target": trg_pop,
                     "synapse_model": self.default_nodes_connection["synapse_model"],
                     "conn_spec": self.default_nodes_connection["conn_spec"],
                     "weight": weight,
                     "delay": lambda source_node, target_node: self.tvb_delay_fun(source_node, target_node),
                     "receptor_type": 0, "source_nodes": src_nodes, "target_nodes": trg_nodes})

        # Creating  devices to be able to observe NEST activity:
        self.output_devices = []
        #          label <- target population
        for pop in self.populations:
            connections = OrderedDict({})
            connections[pop["label"] + "_spikes"] = pop["label"]
            params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"])
            params["record_to"] = self.output_devices_record_to
            self.output_devices.append(
                {"model": "spike_recorder", "params": params,
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to "all"

        # Labels have to be different for every connection to every distinct population
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params.update({"interval": 1.0, "record_to": self.output_devices_record_to,
                       'record_from': ["V_m", "U_m", "I_syn", "I_syn_ex", "I_syn_in", "g_AMPA", "g_GABA_A", "g_L"]})
        for pop in self.populations:
            connections = OrderedDict({})
            #               label    <- target population
            connections[pop["label"]] = pop["label"]
            self.output_devices.append(
                {"model": "multimeter", "params": params,
                 "connections": connections, "nodes": pop["nodes"]})  # None means apply to all

        # Create a spike stimulus input device
        self.input_devices = [
            {"model": "poisson_generator",
             "params": {"rate": self.Estn_stim["rate"], "origin": 0.0, "start": 0.1},
             "connections": {"BaselineEstn": ["E"]},  # "Estn"
             "nodes": self.Estn_nodes_ids,  # None means apply to all
             "weights": self.Estn_stim["weight"], "delays": 0.0, "receptor_type": 1},
            {"model": "poisson_generator",
             "params": {"rate": self.Igpe_stim["rate"], "origin": 0.0, "start": 0.1},
             "connections": {"BaselineIgpe": ["I"]},  # "Igpe"
             "nodes": self.Igpe_nodes_ids,  # None means apply to all
             "weights": self.Igpe_stim["weight"], "delays": 0.0, "receptor_type": 1},
            {"model": "poisson_generator",
             "params": {"rate": self.Igpi_stim["rate"], "origin": 0.0, "start": 0.1},
             "connections": {"BaselineIgpi": ["I"]},  # "Igpi"
             "nodes": self.Igpi_nodes_ids,  # None means apply to all
             "weights": self.Igpi_stim["weight"], "delays": 0.0, "receptor_type": 1},
            # {"model": "ac_generator",
            #  "params": {"frequency": 30.0, "phase": 0.0, "amplitude": 1.0, "offset": 0.0,
            #             "start": 1.0},  # "stop": 100.0  "origin": 0.0,
            #  "connections": {"DBS_Estn": ["E"]},  # "Estn"
            #  "nodes": self.Estn_nodes_ids,  # None means apply to all
            #  "weights": 1.0, "delays": 0.0}
        ]  #

    def paramsE(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsE)
        if node_id in self.Estn_nodes_ids:
            params.update({"a": 0.005, "b": 0.265, "d": 2.0, "I_e": 3.0})
        elif node_id in self.Eth_nodes_ids:
            params.update({"a": 0.02, "b": 0.25, "d": 0.05, "I_e": 3.5})
        return params

    def paramsI(self, node_id):
        # For the moment they are identical, unless you differentiate the noise parameters
        params = deepcopy(self._paramsI)
        if node_id in self.Igpe_nodes_ids:
            params.update({"I_e": 12.0})
        elif node_id in self.Igpi_nodes_ids:
            params.update({"I_e": 30.0})
        return params

    def tvb_delay_fun(self, source_node, target_node):
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

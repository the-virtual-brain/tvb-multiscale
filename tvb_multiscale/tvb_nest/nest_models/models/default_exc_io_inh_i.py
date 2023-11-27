# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import tvb_weight, tvb_delay
from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region  # \
    # random_normal_weight, tvb_weight, random_normal_tvb_weight, \
    # random_uniform_delay, tvb_delay, random_uniform_tvb_delay,

# TODO: Fix errors for random parameters with NEST 3.5!!!


class DefaultExcIOInhIBuilder(NESTNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                      spiking_simulator, config, logger)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale_e = 1
        self.scale_i = 1

        self.w_ee = 1.0
        self.w_ei = 1.0
        self.w_ie = -1.0
        self.w_ii = -1.0
        self.d_ee = self.within_node_delay()
        self.d_ei = self.within_node_delay()
        self.d_ie = self.within_node_delay()
        self.d_ii = self.within_node_delay()

        self.params_E = dict()
        self.params_I = dict()
        self.pop_conns_EE = dict()
        self.pop_conns_EI = dict()
        self.pop_conns_IE = dict()
        self.pop_conns_II = dict()

        self.nodes_conns = dict()

        self.spike_recorder = dict()
        self.multimeter = dict()

        self.spike_stimulus = dict()

    def _params_E(self, node_index):
        return self.params_E

    def _params_I(self, node_index):
        return self.params_I

    def set_E_population(self):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": self._params_E, "scale": self.scale_e, "nodes": None}  # None means "all"
        return pop

    def set_I_population(self):
        pop = {"label": "I", "model": self.default_population["model"],
               "params": self._params_I, "scale": self.scale_i, "nodes": None}  # None means "all"
        return pop

    def set_populations(self):
        # Populations' configurations
        self.populations = [self.set_E_population(), self.set_I_population()]

    # Within region-node connections
    # By default we choose random jitter on weights and delays

    def weight_fun(self, w, scale=1.0, sigma=0.1):
        # return random_normal_weight(w, scale, sigma)
        return w

    def delay_fun(self, low=None, high=None):
        # if low is None:
        #     low = self.default_min_delay
        # if high is None:
        #     high = np.maximum(self.tvb_dt, 2 * self.default_min_delay)
        # return random_uniform_delay(low, low, high, sigma=None)
        return self.default_min_delay

    def within_node_delay(self):
        return self.delay_fun()

    def receptor_E_fun(self):
        return 0

    def receptor_I_fun(self):
        return 0

    def set_EE_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee,  # self.weight_fun(self.w_ee),
             "delay": self.d_ee,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EE)
        return connections

    def set_EI_populations_connections(self):
        connections = \
            {"source": "E", "target": "I",  # E -> I
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ei,  # self.weight_fun(self.w_ei),
             "delay": self.d_ei,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EI)
        return connections

    def set_IE_populations_connections(self):
        connections = \
            {"source": "I", "target": "E",  # I -> E
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ie,  # self.weight_fun(self.w_ie),
             "delay": self.d_ie,
             "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_IE)
        return connections

    def set_II_populations_connections(self):
        connections = \
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ii,  # self.weight_fun(self.w_ii),
             "delay": self.d_ii,
             "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_II)
        return connections

    def set_populations_connections(self):
        self.populations_connections = [
           self.set_EE_populations_connections(), self.set_EI_populations_connections(),
           self.set_IE_populations_connections(), self.set_II_populations_connections()
        ]

    # Among/Between region-node connections
    # By default we choose random jitter around TVB weights and delays

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.global_coupling_scaling
        # return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale, sigma)
        return scale * tvb_weight(source_node, target_node, self.tvb_weights)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        if low is None:
            low = self.default_min_delay
        if high is None:
            high = np.maximum(self.tvb_dt, 2*self.default_min_delay)
        # return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        self.nodes_connections[0].update(self.nodes_conns)

    def set_spike_recorder(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        params = self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
        params["record_to"] = self.output_devices_record_to
        device = {"model": "spike_recorder", "params": params,
                  "neurons_fun": lambda node_id, population: population[:np.minimum(100, len(population))],
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.spike_recorder)
        return device

    def set_multimeter(self):
        connections = OrderedDict()
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        params["record_to"] = self.output_devices_record_to
        device = {"model": "multimeter", "params": params,
                  "neurons_fun": lambda node_id, population: population[:np.minimum(100, len(population))],
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.multimeter)
        return device

    def set_output_devices(self):
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_recorder(), self.set_multimeter()]

    def set_spike_stimulus(self):
        connections = OrderedDict()
        #             label <- target population
        connections["Stimulus"] = ["E"]
        device = \
            {"model": "poisson_generator",
             "params": {"rate": 7000.0, "origin": 0.0, "start": self.spiking_dt},  # "stop": 100.0
             "connections": connections, "nodes": None,
             "weights": 1.0,  # self.weight_fun(1.0),
             # random_uniform_delay(self.default_min_delay, self.default_min_delay, 2*self.default_min_delay, sigma=None),
             "delays": self.default_min_delay,
             "receptor_type": 0}
        device.update(self.spike_stimulus)
        return device

    def set_input_devices(self):
        self.input_devices = [self.set_spike_stimulus()]

    def set_defaults(self):
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        self.set_input_devices()

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(DefaultExcIOInhIBuilder, self).build()


class DefaultExcIOInhIMultisynapseBuilder(DefaultExcIOInhIBuilder):

    model = "aeif_cond_alpha_multisynapse"

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(), spiking_simulator=None,
                 config=CONFIGURED, logger=None):

        super(DefaultExcIOInhIMultisynapseBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                                  spiking_simulator, config, logger)

        self.model = "aeif_cond_alpha_multisynapse"

        self.w_ie = 1.0
        self.w_ii = 1.0

        E_ex = 0.0
        E_in = -85.0
        tau_syn_ex = 0.2
        tau_syn_in = 2.0
        E_rev = np.array([E_ex] +  # exc local spikes
                         [E_in])  # inh local spikes
        tau_syn = np.array([tau_syn_ex] +  # exc spikes
                           [tau_syn_in])  # inh spikes
        self.params_E = {"E_rev": E_rev, "tau_syn": tau_syn}
        self.params_I = self.params_E.copy()

        self.nodes_conns = {"receptor_type": self.receptor_by_source_region_fun}

    def _adjust_multisynapse_params(self, params, multi_params=["E_rev", "tau_syn"]):
        for p in multi_params:
            val = params[p].tolist()
            n_vals = self.number_of_regions - len(val) + 2
            val += [val[0]] * n_vals
            params[p] = np.array(val)
        return params

    def _params_E(self, node_index):
        return self._adjust_multisynapse_params(self.params_E)

    def _params_I(self, node_index):
        return self._adjust_multisynapse_params(self.params_I)

    def receptor_E_fun(self):
        return 1

    def receptor_I_fun(self):
        return 2

    def receptor_by_source_region_fun(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

    def set_defaults(self):
        self.spike_stimulus = {"params": {"rate": 30000.0, "origin": 0.0, "start": self.spiking_dt},  # "stop": 100.0
                               "receptor_type": lambda target_node: target_node + 3}
        super(DefaultExcIOInhIMultisynapseBuilder, self).set_defaults()

    def build(self, set_defaults=True):
        return super(DefaultExcIOInhIMultisynapseBuilder, self).build(set_defaults)

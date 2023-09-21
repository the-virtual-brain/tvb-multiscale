# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

import numpy as np

from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyNetworkBuilder

from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class DefaultExcIOInhIBuilder(ANNarchyNetworkBuilder):

    def __init__(self, tvb_simulator={}, spiking_nodes_inds=[], spiking_simulator=None,
                 config=None, logger=None):
        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator,
                                                      config, logger)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale_e = 1
        self.scale_i = 1

        self.w_ee = 1.0
        self.w_ei = 1.0
        self.w_ie = 1.0
        self.w_ii = 1.0
        self.d_ee = 1.0
        self.d_ei = 1.0
        self.d_ie = 1.0
        self.d_ii = 1.0

        # Parameters following the iaf_cond_alpha NEST model params
        self.params_E = {"v_rest": -70.0, "v_reset": -60.0, "v_thresh": -55.0, "e_rev_E": 0.0, "e_rev_I": -85.0,
                          "cm": 0.25,  # nF
                          "tau_refrac": 2.0, "tau_syn_E": 0.2, "tau_syn_I": 2.0, "tau_m": 16.6667}
        self.params_E = {}
        self.params_I = self.params_E.copy()

        self.pop_conns_EE = {}
        self.pop_conns_EI = {}
        self.pop_conns_IE = {}
        self.pop_conns_II = {}

        self.nodes_conns = {}

        self.spike_monitor = {}
        self.monitor = {}

        self.spike_stimulus = {}

    def set_E_population(self):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": self.params_E, "scale": self.scale_e, "nodes": None}  # None means "all"
        return pop

    def set_I_population(self):
        pop = {"label": "I", "model": self.default_population["model"],
               "params": self.params_I, "scale": self.scale_i, "nodes": None}  # None means "all"
        return pop

    def set_populations(self):
        # Populations' configurations
        self.populations = [self.set_E_population(), self.set_I_population()]

    # Within region-node connections

    def receptor_E_fun(self):
        return "exc"

    def receptor_I_fun(self):
        return "inh"

    def set_EE_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee,
             "delay": self.d_ee,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EE)
        return connections

    def set_EI_populations_connections(self):
        connections = \
            {"source": "E", "target": "I",  # E -> I
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ei,
             "delay": self.d_ei,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EI)
        return connections

    def set_IE_populations_connections(self):
        connections = \
            {"source": "I", "target": "E",  # I -> E
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ie,
             "delay": self.d_ie,
             "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_IE)
        return connections

    def set_II_populations_connections(self):
        connections = \
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ii,
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

    def tvb_weight_fun(self, source_node, target_node, scale=None):
        if scale is None:
            scale = self.global_coupling_scaling
        return scale_tvb_weight(source_node, target_node, self.tvb_weights, scale)

    def tvb_delay_fun(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": "exc", "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        self.nodes_connections[0].update(self.nodes_conns)

    def set_SpikeMonitor(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        device = {"model": "SpikeMonitor", "params": {},
                  "neurons_inds": lambda node_id, neurons_inds:
                                            tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.spike_monitor)
        return device

    def set_Monitor(self):
        connections = OrderedDict()
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = deepcopy(self.config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF["Monitor"])
        params["period"] = self.monitor_period
        device = {"model": "Monitor", "params": deepcopy(params),
                  "neurons_inds": lambda node_id, neurons_inds:
                        tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.monitor)
        return device

    def set_output_devices(self):
        # Creating  devices to be able to observe ANNarchy activity:
        # Labels have to be different
        self.output_devices = [self.set_SpikeMonitor(), self.set_Monitor()]

    def set_spike_stimulus(self):
        connections = OrderedDict()
        #             label <- target population
        connections["Stimulus"] = ["E"]
        device = \
            {"model": "PoissonPopulation",
             "params": {"rates": 6000.0},
             "connections": connections, "nodes": None,
             "weights": 1.0,
             "delays": self.default_min_delay,
             "receptor_type": "exc"}
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

# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyNetworkBuilder


class DefaultExcIOBuilder(ANNarchyNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(DefaultExcIOBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale = 1

        self.w = 1.0
        self.d = 1.0

        # Parameters following the iaf_cond_alpha NEST model params
        self.params = {"v_rest": -70.0, "v_reset": -60.0, "v_thresh": -55.0, "e_rev_E": 0.0, "e_rev_I": -85.0,
                       "cm": 0.25, # nF
                       "tau_refrac": 2.0, "tau_syn_E": 0.2, "tau_syn_I": 2.0, "tau_m": 16.6667}

        self.pop_conns_EE = dict()
        self.nodes_conns = dict()

        self.spike_monitor = dict()
        self.monitor = dict()

        self.spike_stimulus = dict()

    def set_population(self):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": self.params, "scale": self.scale, "nodes": None}  # None means "all"
        return pop

    def set_populations(self):
        # Populations' configurations
        self.populations = [self.set_population()]

    # Within region-node connections

    def receptor_fun(self):
        return "exc"

    def set_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w,
             "delay": self.d,
             "receptor_type": self.receptor_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EE)
        self.populations_connections = [connections]

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
            {"source": "E", "target": ["E"],
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
        return super(DefaultExcIOBuilder, self).build()

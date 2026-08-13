# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import tvb_weight, tvb_delay
from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region  # \
#    random_normal_weight, random_normal_tvb_weight, \
#    random_uniform_delay, random_uniform_tvb_delay,

# TODO: Fix errors for random parameters with NEST 3.5!!!


class DefaultExcIOBuilder(NESTNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(DefaultExcIOBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale = 1

        self.w = 1.0
        self.d = self.within_node_delay()

        self.params = dict()
        self.pop_conns = dict()

        self.nodes_conns = dict()

        self.spike_recorder = dict()
        self.multimeter = dict()

        self.spike_stimulus = dict()

    def set_population(self):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": self.params, "scale": self.scale, "nodes": None}  # None means "all"
        return pop

    def set_populations(self):
        # Populations' configurations
        self.populations = [self.set_population()]

    # Within region-node connections
    # By default we choose random jitter on weights and delays

    def weight_fun(self, w, scale=1.0, sigma=0.1):
        return w
        # return random_normal_weight(w, scale, sigma)

    def delay_fun(self, low=None, high=None):
        # if low is None:
        #     low = self.default_min_delay
        # if high is None:
        #     high = np.maximum(self.tvb_dt, 2 * self.default_min_delay)
        # return random_uniform_delay(low, low, high, sigma=None)
        return self.default_min_delay

    def within_node_delay(self):
        return self.delay_fun()

    def receptor_fun(self):
        return 0

    def set_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.weight_fun(self.w),
             "delay": self.d,
             "receptor_type": self.receptor_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns)
        self.populations_connections = [connections]

    # Among/Between region-node connections
    # By default we choose random jitter around TVB weights and delays

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        # if scale is None:
        #     scale = self.global_coupling_scaling
        # return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale, sigma)
        return tvb_weight(source_node, target_node, self.tvb_weights)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        # if low is None:
        #     low = self.default_min_delay
        # if high is None:
        #     high = np.maximum(self.tvb_dt, 2 * self.default_min_delay)
        # return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)
        return np.maximum(self.default_min_delay, tvb_delay(source_node, target_node, self.tvb_delays))

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": "E",
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        self.nodes_connections[0].update(self.nodes_conns)

    def set_spike_recorder(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"])
        params["record_to"] = self.output_devices_record_to
        device = {"model": "spike_recorder", "params": params,
                  "neurons_fun": lambda node_id, neurons_inds:
                                            tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.spike_recorder)
        return device

    def set_multimeter(self):
        connections = OrderedDict()
        #               label    <- target population
        connections["Excitatory"] = "E"
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
             "params": {"rate": 10000.0, "origin": 0.0, "start": self.spiking_dt},  # "stop": 100.0
             "connections": connections, "nodes": None,
             "weights": self.weight_fun(1.0),
             "delays": self.default_min_delay,  # random_uniform_delay(self.default_min_delay,
                                            # self.default_min_delay, 2*self.default_min_delay, sigma=None),
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
        return super(DefaultExcIOBuilder, self).build()


class DefaultExcIOMultisynapseBuilder(DefaultExcIOBuilder):

    model = "aeif_cond_alpha_multisynapse"

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(), spiking_simulator=None,
                 config=CONFIGURED, logger=None, **kwargs):

        super(DefaultExcIOMultisynapseBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator,
                                                              config, logger)
        self.model = "aeif_cond_alpha_multisynapse"
        E_ex = kwargs.get("E_ex", 0.0)
        E_in = kwargs.get("E_in", -85.0)
        tau_syn_ex = kwargs.get("tau_syn_ex", 0.2)
        tau_syn_in = kwargs.get("tau_syn_in", 2.0)
        E_rev = np.array([E_ex] +  # exc local spikes
                         [E_in])  # inh local spikes
        tau_syn = np.array([tau_syn_ex] +  # exc spikes
                           [tau_syn_in])  # inh spikes
        self.params = {"E_rev": E_rev, "tau_syn": tau_syn}

        self.nodes_conns = {"receptor_type": self.receptor_by_source_region_fun}

    def _adjust_multisynapse_params(self, params, multi_params=["E_rev", "tau_syn"]):
        for p in multi_params:
            val = params[p].tolist()
            val += [val[0]] * self.number_of_regions
            params[p] = np.array(val)
        return params

    def receptor_fun(self):
        return 1

    def receptor_by_source_region_fun(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

    def set_defaults(self):
        self.spike_stimulus = {"params": {"rate": 30000.0, "origin": 0.0, "start": self.spiking_dt},  # "stop": 100.0
                               "receptor_type": lambda target_node: target_node + 3}
        super(DefaultExcIOMultisynapseBuilder, self).set_defaults()

    def build(self, set_defaults=True):
        self.params = self._adjust_multisynapse_params(self.params)
        return super(DefaultExcIOMultisynapseBuilder, self).build(set_defaults)

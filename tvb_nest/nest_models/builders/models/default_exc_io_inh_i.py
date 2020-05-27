# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import random_normal_weight, random_uniform_delay, \
    random_normal_tvb_weight, random_uniform_tvb_delay, receptor_by_source_region


class DefaultExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True):
        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale_e = 1
        self.scale_i = 1

        self.w_ee = self.weight_fun(1.0)
        self.w_ei = self.weight_fun(1.0)
        self.w_ie = self.weight_fun(-1.0)
        self.w_ii = self.weight_fun(-1.0)
        self.d_ee = self.within_node_delay()
        self.d_ei = self.within_node_delay()
        self.d_ie = self.within_node_delay()
        self.d_ii = self.within_node_delay()

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()

        self.params_E = {}
        self.params_I = {}
        self.pop_conns_EE = {}
        self.pop_conns_EI = {}
        self.pop_conns_IE = {}
        self.pop_conns_II = {}

        self.nodes_conns = {}

        self.spike_detector = {}
        self.multimeter = {}

        self.spike_stimulus = {}

        if set_defaults:
            self.set_defaults()

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
    # By default we choose random jitter on weights and delays

    def weight_fun(self, w, scale=1.0, sigma=0.1):
        return random_normal_weight(w, scale, sigma)

    def delay_fun(self, low=None, high=None):
        if low is None:
            low = self.default_populations_connection["delay"]
        if high is None:
            high = np.maximum(self.tvb_dt, 2 * self.default_populations_connection["delay"])
        return random_uniform_delay(low, low, high, sigma=None)

    def within_node_delay(self):
        return self.delay_fun()

    def receptor_E_fun(self):
        return 0

    def receptor_I_fun(self):
        return 0

    def set_EE_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee,
             "delay": self.d_ee,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EE)
        return connections

    def set_EI_populations_connections(self):
        connections = \
            {"source": "E", "target": "I",  # E -> I
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ei,
             "delay": self.d_ei,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_EI)
        return connections

    def set_IE_populations_connections(self):
        connections = \
            {"source": "I", "target": "E",  # I -> E
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ie,
             "delay": self.d_ie,
             "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_IE)
        return connections

    def set_II_populations_connections(self):
        connections = \
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": self.default_populations_connection["model"],
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

    def tvb_weight(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.global_coupling_scaling
        return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale, sigma)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        if low is None:
            low = self.tvb_dt
        if high is None:
            high = 2 * self.tvb_dt
        return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
             "model": self.default_nodes_connection["model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        self.nodes_connections[0].update(self.nodes_conns)

    def set_spike_detector(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        device = {"model": "spike_detector", "params": {},
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.spike_detector)
        return device

    def set_multimeter(self):
        connections = OrderedDict()
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        device = {"model": "multimeter", "params": params,
                  "connections": connections, "nodes": None}  # None means all here
        device.update(self.multimeter)
        return device

    def set_output_devices(self):
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_detector(), self.set_multimeter()]

    def set_spike_stimulus(self):
        connections = OrderedDict()
        #             label <- target population
        connections["Stimulus"] = ["E"]
        device = \
            {"model": "poisson_generator",
             "params": {"rate": 6000.0, "origin": 0.0, "start": 0.1},  # "stop": 100.0
             "connections": connections, "nodes": None,
             "weights": self.weight_fun(1.0),
             "delays": random_uniform_delay(self.tvb_dt, self.tvb_dt, 2*self.tvb_dt, sigma=None),
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


class DefaultExcIOInhIMultisynapseBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):

        super(DefaultExcIOInhIMultisynapseBuilder, self).__init__(
            tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults=False)

        self.default_population["model"] = "aeif_cond_alpha_multisynapse"

        self.w_ie = self.weight_fun(1.0)
        self.w_ii = self.weight_fun(1.0)

        E_ex = kwargs.get("E_ex", 0.0)
        E_in = kwargs.get("E_ex", -85.0)
        tau_syn_ex = kwargs.get("tau_syn_ex", 0.2)
        tau_syn_in = kwargs.get("tau_syn_in", 2.0)
        E_rev = np.array([E_ex] +  # exc local spikes
                         [E_in] +  # inh local spikes
                         self.number_of_nodes * [E_ex])  # ext, exc spikes
        tau_syn = np.array([tau_syn_ex] +  # exc spikes
                           [tau_syn_in] +  # inh spikes
                           self.number_of_nodes * [tau_syn_ex])  # ext, exc spikes
        self.params_E = {"E_rev": E_rev, "tau_rise": tau_syn}
        self.params_I = self.params_E

        self.nodes_conns = {"receptor_type": self.receptor_by_source_region_fun}

        self.spike_stimulus = {"receptor_type": lambda target_node: target_node + 3}

        if set_defaults:
            self.set_defaults()

    def receptor_E_fun(self):
        return 1

    def receptor_I_fun(self):
        return 2

    def receptor_by_source_region_fun(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

    def set_spike_stimulus(self):
        return super(DefaultExcIOInhIMultisynapseBuilder, self).set_spike_stimulus()

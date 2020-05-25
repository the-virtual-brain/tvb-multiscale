# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import random_normal_weight, random_uniform_delay, \
    random_normal_tvb_weight, random_uniform_tvb_delay, receptor_by_source_region


class DefaultExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):
        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        # Common order of neurons' number per population:
        self.population_order = kwargs.pop("population_order", 100)

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()

        self.scale_e = kwargs.pop("scale_e", 1)
        self.scale_i = kwargs.pop("scale_i", 1)
        self.w_ee = kwargs.pop("w_ee", self.weight_fun(1.0))
        self.w_ei = kwargs.pop("w_ei", self.weight_fun(1.0))
        self.w_ie = kwargs.pop("w_ie", self.weight_fun(-1.0))
        self.w_ii = kwargs.pop("w_ii", self.weight_fun(-1.0))
        self.d_ee = kwargs.pop("d_ee", self.within_node_delay())
        self.d_ei = kwargs.pop("d_ei", self.within_node_delay())
        self.d_ie = kwargs.pop("d_ie", self.within_node_delay())
        self.d_ii = kwargs.pop("d_ii", self.within_node_delay())

        if set_defaults:
            self.set_defaults()

    def set_E_population(self, **kwargs):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": {}, "scale": self.scale_e, "nodes": None}  # None means "all"
        pop.update(kwargs)
        return pop

    def set_I_population(self, **kwargs):
        pop = {"label": "I", "model": self.default_population["model"],
               "params": {}, "scale": self.scale_i, "nodes": None}  # None means "all"
        pop.update(kwargs)
        return pop

    def set_populations(self, params_E={}, params_I={}):
        # Populations' configurations
        self.populations = [self.set_E_population(**params_E),
                            self.set_I_population(**params_I)]

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

    def receptor_fun_E(self):
        return 0

    def receptor_fun_I(self):
        return 0

    def set_EE_populations_connections(self, **params):
        connections = \
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee,
             "delay": self.d_ee,
             "receptor_type": self.receptor_fun_E(), "nodes": None}  # None means "all"
        connections.update(params)
        return connections

    def set_EI_populations_connections(self, **params):
        connections = \
            {"source": "E", "target": "I",  # E -> I
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ei,
             "delay": self.d_ei,
             "receptor_type": self.receptor_fun_E(), "nodes": None}  # None means "all"
        connections.update(params)
        return connections

    def set_IE_populations_connections(self, **params):
        connections = \
            {"source": "I", "target": "E",  # I -> E
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ie,
             "delay": self.d_ie,
             "receptor_type": self.receptor_fun_I(), "nodes": None}  # None means "all"
        connections.update(params)
        return connections

    def set_II_populations_connections(self, **params):
        connections = \
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ii,
             "delay": self.d_ii,
             "receptor_type": self.receptor_fun_I(), "nodes": None}  # None means "all"
        connections.update(params)
        return connections

    def set_populations_connections(self, params_EE={}, params_EI={}, params_IE={}, params_II={}):
        self.populations_connections = [
           self.set_EE_populations_connections(**params_EE),
           self.set_EI_populations_connections(**params_EI),
           self.set_IE_populations_connections(**params_IE),
           self.set_II_populations_connections(**params_II)
        ]

    # Among/Between region-node connections
    # By default we choose random jitter around TVB weights and delays

    def tvb_weight(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.global_coupling_scaling
        return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale, sigma)

    def tvb_delay(self, source_node, target_node, low=None, high=None, sigma=0.1):
        if low is None:
            low = self.tvb_dt
        if high is None:
            high = 2 * self.tvb_dt
        return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)

    def receptor_by_source_region(self, source_node=None, target_node=None):
        return 0

    def set_nodes_connections(self, **params):
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
             "model": self.default_nodes_connection["model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight,
             "delay": self.tvb_delay,
             # Each region emits spikes in its own port:
             "receptor_type": self.receptor_by_source_region, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        self.nodes_connections[0].update(params)

    def set_spike_detector(self, connections=OrderedDict({}), **kwargs):
        if len(connections) == 0:
            #          label <- target population
            connections["E"] = "E"
            connections["I"] = "I"
        device = {"model": "spike_detector", "params": {},
                  "connections": connections, "nodes": None}  # None means all here
        device.update(**kwargs)
        return device

    def set_multimeter(self, connections=OrderedDict({}), **kwargs):
        if len(connections) == 0:
            #               label    <- target population
            connections["Excitatory"] = "E"
            connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        device = {"model": "multimeter", "params": params,
                  "connections": connections, "nodes": None}  # None means all here
        device.update(**kwargs)
        return device

    def set_output_devices(self, spike_detector_connections=OrderedDict({}), multimeter_connections=OrderedDict({}),
                           spike_detector_kwargs={}, multimeter_kwargs={}):
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_detector(spike_detector_connections, **spike_detector_kwargs),
                               self.set_multimeter(multimeter_connections, **multimeter_kwargs)]

    def set_spike_stimulus(self, connections=OrderedDict({}), **kwargs):
        if len(connections) == 0:
            #             label <- target population
            connections["Stimulus"] = ["E", "I"]
        device = \
            {"model": "poisson_generator",
             "params": {"rate": 1000.0, "origin": 0.0, "start": 0.1},  # "stop": 100.0
             "connections": connections, "nodes": None,
             "weights": self.weight_fun(1.0), "delays": self.delay_fun(),
             "receptor_types": 0}
        device.update(**kwargs)
        return device

    def set_input_devices(self, spike_stimulus_connections=OrderedDict({}), spike_stimulus_kwargs={}):
        self.input_devices = [self.set_spike_stimulus(spike_stimulus_connections, **spike_stimulus_kwargs)]

    def set_defaults(self):
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        self.set_input_devices()


class DefaultExcIOInhIMultisynapseBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 set_defaults=True, **kwargs):
        self.E_ex = kwargs.get("E_ex", 0.0)
        self.E_in = kwargs.get("E_ex", -85.0)
        self.tau_rise_ex = kwargs.get("tau_rise_ex", 0.2)
        self.tau_rise_in = kwargs.get("tau_rise_in", 2.0)
        self.tau_decay_ex = kwargs.get("tau_decay_ex", 0.2)
        self.tau_decay_in = kwargs.get("tau_decay_in", 2.0)
        self.tau_decay_ex_ext = kwargs.get("tau_decay_ex", 2.0)
        E_rev = np.array([self.E_ex] +  # exc local spikes
                         [self.E_in] +  # inh local spikes
                         self.number_of_nodes * [self.E_ex])  # ext, exc spikes
        tau_rise = np.array([self.tau_rise_ex] +  # exc spikes (AMPA,rec, NMDA)
                            [self.tau_rise_in] +  # inh spikes (GABA)
                            self.number_of_nodes * [self.tau_rise_ex])  # ext, exc spikes (AMPA,ext)
        tau_decay = np.array([self.tau_decay_ex] +  # exc spikes (AMPA,rec, NMDA)
                             [self.tau_decay_in] +  # inh spikes (GABA)
                             self.number_of_nodes * [self.tau_decay_ex_ext])  # ext, exc spikes (AMPA,ext)
        self.multisynapse_populations_params = {"E_rev": E_rev, "tau_rise": tau_rise, "tau_decay": tau_decay}
        kwargs["w_ee"] = kwargs.pop("w_ie", self.weight_fun(1.0))
        kwargs["w_ie"] = kwargs.pop("w_ii", self.weight_fun(1.0))
        super(DefaultExcIOInhIMultisynapseBuilder, self).__init__(
            tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults=False, **kwargs)
        self.default_population["model"] = "aeif_cond_beta_multisynapse"
        if set_defaults:
            self.set_defaults()

    def set_populations(self, params_E={}, params_I={}):
        params_E.update({"params": self.multisynapse_populations_params})
        params_I.update({"params": self.multisynapse_populations_params})
        # Populations' configurations
        self.populations = [self.set_E_population(**params_E),
                            self.set_E_population(**params_I)]

    def receptor_fun_E(self):
        return 1

    def receptor_fun_I(self):
        return 2

    def receptor_by_source_region(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

    def set_spike_stimulus(self, connections=OrderedDict({}), **kwargs):
        kwargs["receptor_types"] = kwargs.pop("receptor_types", lambda target_node: target_node + 3)
        return super(DefaultExcIOInhIMultisynapseBuilder, self).set_spike_stimulus(connections, **kwargs)

# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import tvb_weight, tvb_delay
from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
# from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import \
#     random_normal_weight, random_normal_tvb_weight, \
#     random_uniform_delay, random_uniform_tvb_delay

# TODO: Fix errors for random parameters with NEST 3.5!!!


class JansenRitBuilder(NESTNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(JansenRitBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

        # Common order of neurons' number per population:
        self.population_order = 100

        self.scale_e = 1
        self.scale_ei = 1
        self.scale_ii = 1

        self.w_pe = 1.0
        self.w_pi = 1.0
        self.w_ep = 1.0
        self.w_ip = -1.0
        self.d_pe = self.within_node_delay()
        self.d_pi = self.within_node_delay()
        self.d_ep = self.within_node_delay()
        self.d_ip = self.within_node_delay()

        self.params_E = dict()
        self.params_EI = dict()
        self.params_II = dict()
        self.pop_conns_PE = dict()
        self.pop_conns_PI = dict()
        self.pop_conns_EP = dict()
        self.pop_conns_IP = dict()

        self.nodes_conns = dict()

        self.spike_recorder = dict()
        self.multimeter = dict()

        self.spike_stimulus = dict()

    def _params_E(self, node_index):
        return self.params_E

    def _params_EI(self, node_index):
        return self.params_EI

    def _params_II(self, node_index):
        return self.params_II

    def set_E_population(self):
        pop = {"label": "E", "model": self.default_population["model"],
               "params": self._params_E, "scale": self.scale_e, "nodes": None}  # None means "all"
        return pop

    def set_EI_population(self):
        pop = {"label": "Ein", "model": self.default_population["model"],
               "params": self._params_EI, "scale": self.scale_ei, "nodes": None}  # None means "all"
        return pop

    def set_II_population(self):
        pop = {"label": "Iin", "model": self.default_population["model"],
               "params": self._params_II, "scale": self.scale_ii, "nodes": None}  # None means "all"
        return pop

    def set_populations(self):
        # Populations' configurations
        self.populations = [self.set_E_population(), self.set_EI_population(), self.set_II_population()]

    # Within region-node connections
    # By default we choose random jitter on weights and delays

    def weight_fun(self, w, scale=1.0, sigma=0.1):
        return scale * w

    def within_node_delay(self):
        return self.tvb_dt

    def receptor_E_fun(self):
        return 0

    def receptor_EI_fun(self):
        return 0

    def receptor_II_fun(self):
        return 0

    def set_PE_populations_connections(self):
        connections = \
            {"source": "E", "target": "Ein",  # E -> Ein
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_pe,  # self.weight_fun(self.w_pe),
             "delay": self.d_pe,
             "receptor_type": self.receptor_EI_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_PE)
        return connections

    def set_PI_populations_connections(self):
        connections = \
            {"source": "E", "target": "Iin",  # E -> Iin
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_pi,  # self.weight_fun(self.w_pi),
             "delay": self.d_pi,
             "receptor_type": self.receptor_II_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_PI)
        return connections

    def set_EP_populations_connections(self):
        connections = \
            {"source": "Ein", "target": "E",  # Ein -> E
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ep,  # self.weight_fun(self.w_ep),
             "delay": self.d_ep,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_IP)
        return connections

    def set_IP_populations_connections(self):
        connections = \
            {"source": "Iin", "target": "E",  # Iin -> E
             "synapse_model": self.default_populations_connection["synapse_model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ip,  # self.weight_fun(self.w_ip),
             "delay": self.d_ip,
             "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
        connections.update(self.pop_conns_IP)
        return connections

    def set_populations_connections(self):
        self.populations_connections = [
           self.set_PE_populations_connections(), self.set_PI_populations_connections(),
           self.set_EP_populations_connections(), self.set_IP_populations_connections()
        ]

    # Among/Between region-node connections
    # By default we choose random jitter around TVB weights and delays

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.global_coupling_scaling
        # return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale, sigma)
        return scale * tvb_weight(source_node, target_node, self.tvb_weights)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        # if low is None:
        #     low = self.default_min_delay
        # if high is None:
        #     high = np.maximum(self.tvb_dt, 2 * self.default_min_delay)
        # return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "Ein", "target": "E",
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             "receptor_type": 0, "source_nodes": None, "target_nodes": None},  # None means "all"
            {"source": "Iin", "target": "E",
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": lambda source_node, target_node: -self.tvb_weight_fun(source_node, target_node),
             "delay": self.tvb_delay_fun,
             "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        for ii in range(len(self.nodes_connections)):
            self.nodes_connections[ii].update(self.nodes_conns)

    def set_spike_recorder(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        connections["Ein"] = "Ein"
        connections["Iin"] = "Iin"
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
        connections["E_ts"] = "E"
        connections["Ein_ts"] = "Ein"
        connections["Iin_ts"] = "Iin"
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
             "params": {"rate": 7500.0, "origin": 0.0, "start": self.spiking_dt},  # "stop": 100.0
             "connections": connections, "nodes": None,
             "weights": 1.0,  # self.weight_fun(1.0),
             "delays": self.default_min_delay,
             # random_uniform_delay(self.default_min_delay,
             #                      self.default_min_delay, 2*self.default_min_delay, sigma=None),
             "receptor_type": 0}
        device.update(self.spike_stimulus)
        return device

    def set_input_devices(self):
        self.input_devices = [self.set_spike_stimulus()]

    def set_defaults(self, **kwargs):
        self.w_ee = 10*np.abs(kwargs.get("w_pe",
                                         self.tvb_serial_sim.get("model.a_1",
                                                                 np.array([self.w_pe])))[0].item())
        self.w_ei = 10*np.abs(kwargs.get("w_pi",
                                         self.tvb_serial_sim.get("model.a_3",
                                                                 np.array([self.w_pi])))[0].item())
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        self.set_input_devices()

    def configure(self):
        super(JansenRitBuilder, self).configure()
        self.global_coupling_scaling = \
            self.tvb_serial_sim.get("coupling.cmax", np.array([2.0 * 0.0025]))[0].item() * \
            self.tvb_serial_sim.get("coupling.a", np.array([0.56]))[0].item()

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(JansenRitBuilder, self).build()

# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import tvb_weight, tvb_delay, receptor_by_source_region


class DefaultExcIOInhIMultisynapseBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 **kwargs):
        config.nest.DEFAULT_MODEL = "aeif_cond_beta_multisynapse"
        super(DefaultExcIOInhIMultisynapseBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        self.scale_e = kwargs.get("scale_e", 1)
        self.scale_i = kwargs.get("scale_i", 1)
        self.w_ee = kwargs.get("w_ee", 1.0)
        self.w_ei = kwargs.get("w_ei", 1.0)
        self.w_ie = kwargs.get("w_ie", -1.0)
        self.w_ii = kwargs.get("w_ii", -1.0)
        self.E_ex = kwargs.get("E_ex", 0.0)
        self.E_in = kwargs.get("E_ex", -85.0)
        self.tau_rise_ex = kwargs.get("tau_rise_ex", 0.2)
        self.tau_rise_in = kwargs.get("tau_rise_in", 2.0)
        self.tau_decay_ex = kwargs.get("tau_decay_ex", 0.2)
        self.tau_decay_in = kwargs.get("tau_decay_in", 2.0)
        self.tau_decay_ex_ext = kwargs.get("tau_decay_ex", 2.0)

        # Common order of neurons' number per population:
        self.population_order = 100

        E_rev = np.array([self.E_ex] +  # exc local spikes
                         [self.E_in] +  # inh local spikes
                         self.number_of_nodes * [self.E_ex])  # ext, exc spikes
        tau_rise = np.array([self.tau_rise_ex] +  # exc spikes (AMPA,rec, NMDA)
                            [self.tau_rise_in] +  # inh spikes (GABA)
                            self.number_of_nodes * [self.tau_rise_ex])  # ext, exc spikes (AMPA,ext)
        tau_decay = np.array([self.tau_decay_ex] +  # exc spikes (AMPA,rec, NMDA)
                             [self.tau_decay_in] +  # inh spikes (GABA)
                             self.number_of_nodes * [self.tau_decay_ex_ext])  # ext, exc spikes (AMPA,ext)
        params = {"E_rev": E_rev, "tau_rise": tau_rise, "tau_decay": tau_decay}
        # Populations' configurations
        self.populations = [{"label": "E", "model": self.default_population["model"],
                             "params": params, "scale": self.scale_e,  "nodes": None},  # None means "all"
                            {"label": "I", "model": self.default_population["model"],
                             "params": params, "scale": self.scale_i, "nodes": None}  # None means "all"
                            ]

        # Within region-node connections
        self.populations_connections = [
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": np.abs(self.w_ee),
             "delay": self.default_populations_connection["delay"],
             "receptor_type": 1, "nodes": None},  # None means "all"
             {"source": "E", "target": "I",  # E -> I
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": np.abs(self.w_ei),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 1, "nodes": None},  # None means "all"
             {"source": "I", "target": "E",  # I -> E
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": np.abs(self.w_ie),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 2, "nodes": None},  # None means "all"
             {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight":np.abs(self.w_ii),
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 2, "nodes": None},  # None means "all"
             ]

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": self.tvb_weight,
              "delay": self.tvb_delay,
              # Each region emits spikes in its own port:
              "receptor_type": self.receptor_by_source_region,
             "source_nodes": None, "target_nodes": None}  # None means "all"
                                 ]

        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = []
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        self.output_devices.append({"model": "spike_detector", "params": {},
                                    "connections": connections, "nodes": None})  # None means all here
        connections = OrderedDict({})
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means all here

    def tvb_weight(self, source_node, target_node):
        return tvb_weight(source_node, target_node, self.tvb_weights)

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_by_source_region(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

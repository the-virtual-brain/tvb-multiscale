# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIMultisynapseBuilder
from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay
from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class WWDeco2013Builder(DefaultExcIOInhIMultisynapseBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):

        super(WWDeco2013Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                set_defaults=False)

        self.default_population["model"] = "iaf_cond_ww_deco"

        self.scale_e = 1.0
        self.scale_i = 1.0

        self.V_th = -50.0  # mV
        self.V_reset = -55.0  # mV
        self.E_L = -70.0  # mV
        self.E_ex = 0.0  # mV
        self.E_in = -70.0  # mV
        # exc neurons (AMPA,rec/ext, NMDA)
        self.C_m_ex = 500.0  # pF
        self.g_L_ex = 25.0  # nS
        self.t_ref_ex = 2.0  # ms
        self.g_AMPA_ext_ex = 2.496  # nS
        self.g_AMPA_rec_ex = 0.104  # nS
        self.g_NMDA_ex = 0.327  # nS
        self.g_GABA_ex = 4.375  # nS
        # inh neurons (GABA):
        self.C_m_in = 200.0  # pF
        self.g_L_in = 20.0  # nS
        self.t_ref_in = 1.0  # ms
        self.g_AMPA_ext_in = 1.944  # nS
        self.g_AMPA_rec_in = 0.081  # nS
        self.g_NMDA_in = 0.258  # nS
        self.g_GABA_in = 3.4055  # nS
        # exc spikes (AMPA,rec/ext, NMDA):
        self.tau_decay_AMPA = 2.0  # ms
        self.tau_decay_NMDA = 100.0  # ms
        self.tau_rise_NMDA = 2.0  # ms
        # inh spikes (GABA):
        self.tau_decay_GABA = 10.0  # ms
        self.w_EE = 1.55
        self.w_IE = 1.0
        self.w_EI = 1.0
        self.w_II = 1.0
        self.stimulus_spike_rate = 2180.0  # Hz

        # For this model we don't want any randomness in connections
        self.w_ee = 1.0
        self.w_ei = 1.0
        self.w_ie = -1.0
        self.w_ii = -1.0

        self.d_ee = self.spiking_dt
        self.d_ie = self.spiking_dt
        self.d_ei = self.spiking_dt
        self.d_ii = self.spiking_dt

        self.global_coupling_scaling *= self.tvb_model.G[0].item()
        self.lamda = 0.0

        if set_defaults:
            self.set_defaults()

    def set_defaults(self):

        # Populations' configurations

        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.N_E = int(self.population_order * self.scale_e)
        self.N_I = int(self.population_order * self.scale_i)

        common_params = {
            "V_th": self.V_th, "V_reset": self.V_reset, "E_L": self.E_L, "E_ex": self.E_ex, "E_in": self.E_in,
            "tau_decay_AMPA": self.tau_decay_AMPA, "tau_decay_GABA_A": self.tau_decay_GABA,
            "tau_decay_NMDA": self.tau_decay_NMDA, "tau_rise_NMDA": self.tau_rise_NMDA,
            "s_AMPA_ext_max": self.N_E * np.ones((self.number_of_nodes,)).astype("f"),
            "N_E": self.N_E, "N_I": self.N_I
        }
        params_E = {
                    "C_m": self.C_m_ex, "g_L": self.g_L_ex, "t_ref": self.t_ref_ex,
                    "g_AMPA_ext": self.g_AMPA_ext_ex, "g_AMPA": self.g_AMPA_rec_ex,
                    "g_NMDA": self.g_NMDA_ex, "g_GABA_A": self.g_GABA_ex,
                    "w_E": self.w_EE, "w_I": self.w_IE
                    }
        params_E.update(common_params)
        self.params_E = lambda node_index: self.param_fun(node_index, params_E,
                                                          weight=self.global_coupling_scaling)
        params_I = {
                    "C_m": self.C_m_in, "g_L": self.g_L_in, "t_ref": self.t_ref_in,
                    "g_AMPA_ext": self.g_AMPA_ext_in, "g_AMPA": self.g_AMPA_rec_in,
                    "g_NMDA": self.g_NMDA_in, "g_GABA_A": self.g_GABA_in,
                    "w_E": self.w_EI, "w_I": self.w_II
                    }
        params_I.update(common_params)
        self.params_I = lambda node_index: self.param_fun(node_index, params_I,
                                                          weight=self.lamda * self.global_coupling_scaling)

        self.nodes_conns_EE = {"weight": 1.0}
        self.nodes_conns_EI = {"weight": 1.0}

        record_from = ["V_m",
                       "s_AMPA", "x_NMDA", "s_NMDA", "s_GABA",
                       "I_AMPA", "I_NMDA", "I_GABA", "I_L", "I_e",
                       "spikes_exc", "spikes_inh"]
        for i_node in range(self.number_of_nodes):
            record_from.append("s_AMPA_ext_%d" % i_node)
            record_from.append("I_AMPA_ext_%d" % i_node)
            record_from.append("spikes_exc_ext_%d" % i_node)
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["record_from"] = record_from
        params["record_to"] = self.output_devices_record_to
        self.multimeter["params"] = params
        self.spike_stimulus = {"params": {"rate": self.stimulus_spike_rate, "origin": 0.0, "start": 0.1},
                               "connections": {"Stimulus": ["E", "I"]},
                               "weights": 1.0, "delays": self.tvb_dt,
                               "receptor_type": lambda target_node: target_node + 1}
        super(WWDeco2013Builder, self).set_defaults()

    def param_fun(self, node_index, params, weight):
        w_E_ext = weight * self.tvb_weights[:, node_index]
        w_E_ext[node_index] = 1.0  # this is external input weight to this node
        out_params = deepcopy(params)
        out_params.update({"w_E_ext": w_E_ext})
        return out_params

    def receptor_E_fun(self):
        return 0

    def receptor_I_fun(self):
        return 0

    def tvb_weight_fun(self, source_node, target_node):
        return 1.0

    def tvb_delay_fun(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_by_source_region_fun(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=1)

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": ["E"],
             "synapse_model": self.default_nodes_connection["synapse_model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": 1.0,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": self.receptor_by_source_region_fun,
             "source_nodes": None, "target_nodes": None}
            # None means "all"
        ]
        self.nodes_connections[0].update(self.nodes_conns_EE)
        if self.lamda:
            self.nodes_connections.append(
                {"source": "E", "target": ["I"],
                 "synapse_model": self.default_nodes_connection["synapse_model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": 1.0,
                 "delay": self.tvb_delay_fun,
                 # Each region emits spikes in its own port:
                 "receptor_type": self.receptor_by_source_region_fun,
                 "source_nodes": None, "target_nodes": None}
            )
            self.nodes_connections[1].update(self.nodes_conns_EI)


class WWDeco2014Builder(WWDeco2013Builder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):

        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                set_defaults=False)

        self.default_population["model"] = "iaf_cond_ww_deco"

        self.scale_e = 1.6
        self.scale_i = 0.4

        self.g_AMPA_ext_ex = 3.37  # nS
        self.g_AMPA_rec_ex = 0.065  # nS
        self.g_NMDA_ex = 0.20  # nS
        self.g_GABA_ex = 10.94  # nS
        # inh neurons (GABA):
        self.g_AMPA_ext_in = 2.59  # nS
        self.g_AMPA_rec_in = 0.051  # nS
        self.g_NMDA_in = 0.16  # nS
        self.g_GABA_in = 8.51  # nS
        self.stimulus_spike_rate = 2400.0  # Hz

        self.lamda = self.tvb_model.lamda[0].item()
        self.w_EE = kwargs.get("w_EE",
                          kwargs.get("w_p",
                                     getattr(self.tvb_model, "w_p", np.array([1.4, ]))[0].item()))
        self.w_IE = kwargs.get("w_IE",
                          kwargs.get("J_i",
                                     getattr(self.tvb_model, "J_i", np.array([1.0, ]))[0].item()))

        if set_defaults:
            WWDeco2013Builder.set_defaults(self)

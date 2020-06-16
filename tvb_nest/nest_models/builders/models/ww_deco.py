# -*- coding: utf-8 -*-

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIMultisynapseBuilder
from tvb_multiscale.spiking_models.builders.templates import tvb_delay, receptor_by_source_region


class WWDeco2013Builder(DefaultExcIOInhIMultisynapseBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 V_th=-50.0,  # mV
                 V_reset=-55.0,  # mV
                 E_L=-70.0,  # mV
                 E_ex=0.0,  # mV
                 E_in=-70.0,  # mV
                 # exc neurons (AMPA,rec/ext, NMDA)
                 C_m_ex=500.0,  # pF
                 g_L_ex=25.0,  # nS
                 t_ref_ex=2.0,  # ms
                 g_AMPA_ext_ex=2.496,  # nS
                 g_AMPA_rec_ex=0.104,  # nS
                 g_NMDA_ex=0.327,  # nS
                 g_GABA_ex=4.375,  # nS
                 # inh neurons (GABA):
                 C_m_in=200.0,  # pF
                 g_L_in=20.0,  # nS
                 t_ref_in=1.0,  # ms
                 g_AMPA_ext_in=1.944,  # nS
                 g_AMPA_rec_in=0.081,  # nS
                 g_NMDA_in=0.258,  # nS
                 g_GABA_in=3.4055,  # nS
                 # exc spikes (AMPA,rec/ext, NMDA):
                 tau_decay_AMPA=2.0,    # ms
                 tau_decay_NMDA=100.0,  # ms
                 tau_rise_NMDA=2.0,  # ms
                 # inh spikes (GABA):
                 tau_decay_GABA=10.0,  # ms
                 w_EE=1.55, w_IE=1.0,
                 stimulus_spike_rate=2180.0,  # Hz
                 **kwargs
                 ):

        super(WWDeco2013Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                set_defaults=False)

        self.default_population["model"] = "iaf_cond_ww_deco"

        self.scale_e = 1.0
        self.scale_i = 1.0

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
            self._set_defaults(V_th, V_reset, E_L, E_ex, E_in,
                               C_m_ex, g_L_ex, t_ref_ex,
                               g_AMPA_ext_ex, g_AMPA_rec_ex, g_NMDA_ex, g_GABA_ex,  # nS
                               C_m_in, g_L_in, t_ref_in,  # ms
                               g_AMPA_ext_in, g_AMPA_rec_in, g_NMDA_in, g_GABA_in,  # nS
                               tau_decay_AMPA, tau_decay_NMDA, tau_rise_NMDA, tau_decay_GABA,  # ms
                               w_EE, w_IE, stimulus_spike_rate)

    def _set_defaults(self,
                      V_th=-50.0,  # mV
                      V_reset=-55.0,  # mV
                      E_L=-70.0,  # mV
                      E_ex=0.0,  # mV
                      E_in=-70.0,  # mV
                      # exc neurons (AMPA,rec/ext, NMDA)
                      C_m_ex=500.0,  # pF
                      g_L_ex=25.0,  # nS
                      t_ref_ex=2.0,  # ms
                      g_AMPA_ext_ex=2.496,  # nS
                      g_AMPA_rec_ex=0.104,  # nS
                      g_NMDA_ex=0.327,  # nS
                      g_GABA_ex=4.375,  # nS
                      # inh neurons (GABA):
                      C_m_in=200.0,  # pF
                      g_L_in=20.0,  # nS
                      t_ref_in=1.0,  # ms
                      g_AMPA_ext_in=1.944,  # nS
                      g_AMPA_rec_in=0.081,  # nS
                      g_NMDA_in=0.258,  # nS
                      g_GABA_in=3.4055,  # nS
                      # exc spikes (AMPA,rec/ext, NMDA):
                      tau_decay_AMPA=2.0,    # ms
                      tau_decay_NMDA=100.0,  # ms
                      tau_rise_NMDA=2.0,  # ms
                      # inh spikes (GABA):
                      tau_decay_GABA=10.0,  # ms
                      w_EE=1.55, w_IE=1.0,
                      stimulus_spike_rate=2180.0  # Hz
                      ):

        # Populations' configurations

        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        N_E = int(self.population_order * self.scale_e)
        N_I = int(self.population_order * self.scale_i)

        common_params = {
            "V_th": V_th, "V_reset": V_reset, "E_L": E_L, "E_ex": E_ex, "E_in": E_in,
            "tau_decay_AMPA": tau_decay_AMPA, "tau_decay_GABA_A": tau_decay_GABA,
            "tau_decay_NMDA": tau_decay_NMDA, "tau_rise_NMDA": tau_rise_NMDA,
            "s_AMPA_ext_max": N_E * np.ones((self.number_of_nodes,)).astype("f")
        }
        params_E = dict(common_params)
        params_E.update({
            "C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex,
            "g_AMPA_ext": g_AMPA_ext_ex, "g_AMPA": g_AMPA_rec_ex,
            "g_NMDA": g_NMDA_ex, "g_GABA_A": g_GABA_ex,
            "w_E": w_EE, "w_I": w_IE,
            "N_E": N_E, "N_I": N_I
        })
        self.params_E = lambda node_index: self.param_fun(node_index, params_E,
                                                          weight=self.global_coupling_scaling)
        params_I = dict(common_params)
        params_I.update({
            "C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in,
            "g_AMPA_ext": g_AMPA_ext_in, "g_AMPA": g_AMPA_rec_in,
            "g_NMDA": g_NMDA_in, "g_GABA_A": g_GABA_in,
            "w_E": 1.0, "w_I": 1.0,
            "N_E": N_E, "N_I": N_I
        })
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
        self.multimeter["params"] = params
        self.spike_stimulus = {"params": {"rate": stimulus_spike_rate, "origin": 0.0, "start": 0.1},
                               "connections": {"Stimulus": ["E", "I"]},
                               "weights": 1.0, "delay": self.tvb_dt,
                               "receptor_type": lambda target_node: target_node + 1}
        self.set_defaults()

    def param_fun(self, node_index, params, weight):
        w_E_ext = weight * self.tvb_weights[:, list(self.spiking_nodes_ids).index(node_index)]
        w_E_ext[node_index] = 1.0  # this is external input weight to this node
        out_params = dict(params)
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
             "model": self.default_nodes_connection["model"],
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
                 "model": self.default_nodes_connection["model"],
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
                 V_th=-50.0,  # mV
                 V_reset=-55.0,  # mV
                 E_L=-70.0,  # mV
                 E_ex=0.0,  # mV
                 E_in=-70.0,  # mV
                 # exc neurons (AMPA,rec/ext, NMDA)
                 C_m_ex=500.0,  # pF
                 g_L_ex=25.0,  # nS
                 t_ref_ex=2.0,  # ms
                 g_AMPA_ext_ex=3.37,  # nS
                 g_AMPA_rec_ex=0.065,  # nS
                 g_NMDA_ex=0.20,  # nS
                 g_GABA_ex=10.94,  # nS
                 # inh neurons (GABA):
                 C_m_in=200.0,  # pF
                 g_L_in=20.0,  # nS
                 t_ref_in=1.0,  # ms
                 g_AMPA_ext_in=2.59,  # nS
                 g_AMPA_rec_in=0.051,  # nS
                 g_NMDA_in=0.16,  # nS
                 g_GABA_in=8.51,  # nS
                 # exc spikes (AMPA,rec/ext, NMDA):
                 tau_decay_AMPA=2.0,    # ms
                 tau_decay_NMDA=100.0,  # ms
                 tau_rise_NMDA=2.0,  # ms
                 # inh spikes (GABA):
                 tau_decay_GABA=10.0,  # ms
                 stimulus_spike_rate=2400.0,  # Hz
                 **kwargs
                 ):

        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                set_defaults=False)

        self.default_population["model"] = "iaf_cond_ww_deco"

        self.scale_e = 1.6
        self.scale_i = 0.4

        self.lamda = self.tvb_model.lamda[0].item()
        w_EE = kwargs.get("w_EE", kwargs.get("w_p", getattr(self.tvb_model, "w_p", 1.4)))
        w_IE = kwargs.get("w_IE", kwargs.get("J_i", getattr(self.tvb_model, "J_i", 1.0)))

        if set_defaults:
            self._set_defaults(V_th, V_reset, E_L, E_ex, E_in,
                               C_m_ex, g_L_ex, t_ref_ex,
                               g_AMPA_ext_ex, g_AMPA_rec_ex, g_NMDA_ex, g_GABA_ex,  # nS
                               C_m_in, g_L_in, t_ref_in,  # ms
                               g_AMPA_ext_in, g_AMPA_rec_in, g_NMDA_in, g_GABA_in,  # nS
                               tau_decay_AMPA, tau_decay_NMDA, tau_rise_NMDA, tau_decay_GABA,  # ms
                               w_EE, w_IE, stimulus_spike_rate)

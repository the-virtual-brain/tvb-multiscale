# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder
from tvb_multiscale.spiking_models.builders.templates import receptor_by_source_region


class RedWWExcIOInhIBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 V_th=-50.0,  # mV
                 V_reset=-55.0,  # mV
                 E_L=-70.0,  # mV
                 # exc neurons (AMPA,rec/ext, NMDA)
                 C_m_ex=500.0,  # pF
                 g_L_ex=25.0,  # nS
                 t_ref_ex=2.0,  # ms
                 # inh neurons (GABA):
                 C_m_in=200.0,  # pF
                 g_L_in=20.0,  # nS
                 t_ref_in=1.0,  # ms
                 # exc spikes (AMPA,rec/ext, NMDA):
                 E_ex=0.0,  # mV
                 tau_decay_ex=100.0,  # maximum(AMPA,rec, NMDA) = maximum(2.0, 100.0) ms
                 tau_rise_ex=2.0,  # tau_rise_NMDA = 2.0 ms
                 # ext, exc spikes(AMPA, ext):
                 # inh spikes (GABA):
                 E_in=-70.0,  # mV
                 tau_decay_in=10.0,  # tau_GABA = 10.0 ms
                 tau_rise_in=1.0,  # assuming tau_rise_GABA = 1.0 ms
                 **kwargs):

        # Populations' configurations
        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property

        common_params = {
            "V_th": V_th, "V_reset": V_reset, "E_L": E_L,
            "E_ex": E_ex, "E_in": E_in,
            "tau_rise_ex": tau_rise_ex, "tau_rise_in": tau_rise_in,
            "tau_decay_ex": tau_decay_ex, "tau_decay_in": tau_decay_in,
        }

        self.params_ex = dict(common_params)
        self.params_ex.update({
            "C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex,
        })
        self.params_in = dict(common_params)
        self.params_in.update({
            "C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in,
        })

        kwargs["scale_e"] = kwargs.pop("scale_e", 1.6)
        kwargs["scale_i"] = kwargs.pop("scale_i", 0.4)

        # config.DEFAULT_MODEL = "iaf_cond_deco2014"
        super(RedWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                    set_defaults=False, **kwargs)
        self.w_ee = kwargs.pop("w_ee", self.weight_fun(self.tvb_model.w_p[0].item()))
        self.w_ie = kwargs.pop("w_ie", self.weight_fun(-self.tvb_model.J_i[0].item()))
        self.global_coupling_scaling *= self.tvb_model.G[0].item()
        self.lamda = self.tvb_model.lamda[0].item()

        if set_defaults:
            self.set_defaults()

    def tvb_weight_exc_fun(self, source_node, target_node):
        return super(RedWWExcIOInhIBuilder, self).tvb_weight(source_node, target_node,
                                                             scale=self.global_coupling_scaling)

    def tvb_weight_inh_fun(self, source_node, target_node):
        return super(RedWWExcIOInhIBuilder, self).tvb_weight(source_node, target_node,
                                                             scale=self.lamda*self.global_coupling_scaling)

    def set_nodes_connections(self, params_EE={}, params_EI={}):
        self.nodes_connections = [
            {"source": "E", "target": ["E"],
             "model": self.default_nodes_connection["model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.tvb_weight_exc_fun,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": self.receptor_by_source_region_fun,
             "source_nodes": None, "target_nodes": None}
            # None means "all"
        ]
        self.nodes_connections[0].update(params_EE)
        if self.lamda:
            self.nodes_connections.append(
                {"source": "E", "target": ["I"],
                 "model": self.default_nodes_connection["model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": self.tvb_weight_inh_fun,
                 "delay": self.tvb_delay_fun,
                 # Each region emits spikes in its own port:
                 "receptor_type": self.receptor_by_source_region_fun,
                 "source_nodes": None, "target_nodes": None}
            )
            self.nodes_connections[1].update(params_EI)

    def set_spike_stimulus(self, connections=OrderedDict({}), **kwargs):
        kwargs["params"] = kwargs.pop("params", {"rate": 6100.0, "origin": 0.0, "start": 0.1})
        return super(RedWWExcIOInhIBuilder, self).set_spike_stimulus(connections, **kwargs)


class RedWWExcIOInhIMultisynapseBuilder(RedWWExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 V_th=-50.0,            # mV
                 V_reset=-55.0,         # mV
                 E_L=-70.0,             # mV
                 tau_w=1000.0,          # ms a very large tau_w to practically eliminate spike adaptation current
                 # exc neurons (AMPA,rec/ext, NMDA)
                 C_m_ex=500.0,          # pF
                 g_L_ex=25.0,           # nS
                 t_ref_ex=2.0,          # ms
                 # inh neurons (GABA):
                 C_m_in=200.0,          # pF
                 g_L_in=20.0,           # nS
                 t_ref_in=1.0,          # ms
                 # exc spikes (AMPA,rec/ext, NMDA):
                 E_ex=0.0,              # mV
                 tau_rise_ex=2.0,      # tau_rise_NMDA =2.0 ms
                 tau_decay_ex=100.0,   # maximum(AMPA,rec, NMDA) = maximum(2.0, 100.0) ms
                 # ext, exc spikes(AMPA, ext):
                 tau_decay_ex_ext=2.0,       # ms
                 # inh spikes (GABA):
                 E_in=-70.0,            # mV
                 tau_rise_in=1.0,   # assuming tau_rise_GABA = 1.0 ms
                 tau_decay_in=10.0,  # tau_decay_GABA =10.0 ms
                 **kwargs):

        super(RedWWExcIOInhIMultisynapseBuilder, self).__init__(
            tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults=False, **kwargs)
        self.default_population["model"] = "aeif_cond_beta_multisynapse"
        self.w_ee = kwargs.pop("w_ee", self.weight_fun(self.tvb_model.w_p[0].item()))
        self.w_ie = kwargs.pop("w_ie", self.weight_fun(self.tvb_model.J_i[0].item()))
        self.w_ii = kwargs.pop("w_ii", self.weight_fun(1.0))
        # Populations' configurations
        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property

        E_rev = np.array([E_ex] +  # exc spikes (AMPA,rec, NMDA)
                         [E_in] +  # inh spikes (GABA)
                         self.number_of_nodes * [E_ex])  # ext, exc spikes (AMPA,ext)
        tau_rise = np.array([tau_rise_ex] +  # exc spikes (AMPA,rec, NMDA)
                            [tau_rise_in] +  # inh spikes (GABA)
                            self.number_of_nodes * [tau_rise_ex])  # ext, exc spikes (AMPA,ext)
        tau_decay = np.array([tau_decay_ex] +  # exc spikes (AMPA,rec, NMDA)
                            [tau_decay_in] +  # inh spikes (GABA)
                            self.number_of_nodes * [tau_decay_ex_ext])  # ext, exc spikes (AMPA,ext)
        common_params = {
            "V_th": V_th, "V_peak": 0.0, "V_reset": V_reset, "E_L": E_L, "E_rev": E_rev,
            "tau_w": tau_w, "tau_rise": tau_rise, "tau_decay": tau_decay}
        self.params_ex = dict(common_params)
        self.params_ex.update({"C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex})
        self.params_in = dict(common_params)
        self.params_in.update({"C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in})

        if set_defaults:
            self.set_defaults()

    def set_populations(self, params_E={}, params_I={}):
        params_E.update({"params": self.params_ex})
        params_I.update({"params": self.params_in})
        # Populations' configurations
        self.populations = [self.set_E_population(**params_E),
                            self.set_I_population(**params_I)]

    def receptor_E_fun(self):
        return 1

    def receptor_I_fun(self):
        return 2

    def receptor_by_source_region_fun(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)

    def set_spike_stimulus(self, connections=OrderedDict({}), **kwargs):
        kwargs["params"] = kwargs.pop("params", {"rate": 1100.0, "origin": 0.0, "start": 0.1})
        kwargs["receptor_types"] = kwargs.pop("receptor_types", lambda target_node: target_node + 3)
        return super(RedWWExcIOInhIMultisynapseBuilder, self).set_spike_stimulus(connections, **kwargs)

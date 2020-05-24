# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIMultisynapseBuilder
from tvb_multiscale.spiking_models.builders.templates import tvb_delay, receptor_by_source_region


class WWDeco2014Builder(RedWWExcIOInhIMultisynapseBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 V_th=-50.0,  # mV
                 V_reset=-55.0,  # mV
                 E_L=-70.0,  # mV
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
                 E_ex=0.0,  # mV
                 tau_decay_AMPA=2.0,    # ms
                 tau_decay_NMDA=100.0,  # ms
                 tau_rise_NMDA=2.0,  # ms
                 # inh spikes (GABA):
                 E_in=-70.0,  # mV
                 tau_decay_GABA=10.0,  # ms
                 exc_pop_scale=1.6,
                 inh_pop_scale=0.4,
                 **kwargs
                 ):

        # For this model we don't want any randomness in connections
        kwargs["w_ee"] = kwargs.pop("w_ee", 1.0)
        kwargs["w_ei"] = kwargs.pop("w_ei", 1.0)
        kwargs["w_ie"] = kwargs.pop("w_ie", -1.0)
        kwargs["w_ii"] = kwargs.pop("w_ii", -1.0)
        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config,
                                                set_defaults=False, **kwargs)
        self.default_population["model"] = "iaf_cond_deco2014"
        self.d_ee = kwargs.pop("d_ee", self.default_populations_connection["delay"])
        self.d_ie = kwargs.pop("d_ie", self.default_populations_connection["delay"])
        self.d_ei = kwargs.pop("d_ei", self.default_populations_connection["delay"])
        self.d_ii = kwargs.pop("d_ii", self.default_populations_connection["delay"])

        # Populations' configurations
        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        exc_pop_size = int(self.population_order * exc_pop_scale)
        inh_pop_size = int(self.population_order * inh_pop_scale)

        common_params = {
            "V_th": V_th, "V_reset": V_reset, "E_L": E_L,  "E_ex": E_ex,  "E_in": E_in,
            "tau_decay_AMPA": tau_decay_AMPA, "tau_decay_GABA_A": tau_decay_GABA,
            "tau_decay_NMDA": tau_decay_NMDA, "tau_rise_NMDA": tau_rise_NMDA,
            "s_AMPA_ext_max": exc_pop_size*np.ones((self.number_of_nodes, )).astype("f")
        }
        self.params_ex = dict(common_params)
        self.params_ex.update({
            "C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex,
            "g_AMPA_ext": g_AMPA_ext_ex, "g_AMPA": g_AMPA_rec_ex,
            "g_NMDA": g_NMDA_ex, "g_GABA_A": g_GABA_ex,
            "w_E": self.tvb_model.w_p[0], "w_I": self.tvb_model.J_i[0],
            "N_E": exc_pop_size-1, "N_I": inh_pop_size  # assuming self connections are not allowed
        })
        self.params_in = dict(common_params)
        self.params_in.update({
            "C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in,
            "g_AMPA_ext": g_AMPA_ext_in, "g_AMPA": g_AMPA_rec_in,
            "g_NMDA": g_NMDA_in, "g_GABA_A": g_GABA_in,
            "w_E": 1.0, "w_I": 1.0,
            "N_E": exc_pop_size, "N_I": inh_pop_size-1  # assuming self connections are not allowed
        })

        if set_defaults:
            self.set_defaults()

    def param_fun(self, node_index, params, weight):
        w_E_ext = weight * self.tvb_weights[:, list(self.spiking_nodes_ids).index(node_index)]
        w_E_ext[node_index] = 1.0  # this is external input weight to this node
        out_params = dict(params)
        out_params.update({"w_E_ext": w_E_ext})
        return out_params

    def set_populations(self, params_E={}, params_I={}):
        paramsE = {"params": lambda node_index: self.param_fun(node_index, self.params_ex,
                                                               weight=self.weight_exc)}
        paramsE.update(params_E)
        paramsI = {"params": lambda node_index: self.param_fun(node_index, self.params_in,
                                                               weight=self.weight_inh)}
        paramsE.update(params_I)
        # Populations' configurations
        self.populations = [self.set_E_population(**paramsE),
                            self.set_I_population(**paramsI)]

    def receptor_fun_E(self):
        return 0

    def receptor_fun_I(self):
        return 0

    def tvb_weight_exc(self, source_node, target_node):
        return 1.0

    def tvb_weight_inh(self, source_node, target_node):
        return 1.0

    def set_nodes_connections(self, params_EE={}, params_EI={}):
        params_EE.update({"weight": 1.0})
        params_EI.update({"weight": 1.0})
        super(WWDeco2014Builder, self).set_nodes_connections(params_EE, params_EI)

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_by_source_region(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=1)

    def set_multimeter(self, connections=OrderedDict({}), **kwargs):
        params = kwargs.pop("params", dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]))
        params['record_from'] = ["V_m",
                                 "s_AMPA", "x_NMDA", "s_NMDA", "s_GABA",
                                 "I_AMPA", "I_NMDA", "I_GABA", "I_L", "I_e",
                                 "spikes_exc", "spikes_inh"
                                 ]
        for i_node in range(self.number_of_nodes):
            params['record_from'].append("s_AMPA_ext_%d" % i_node)
            params['record_from'].append("I_AMPA_ext_%d" % i_node)
            params['record_from'].append("spikes_exc_ext_%d" % i_node)
        kwargs["params"] = params
        return super(WWDeco2014Builder, self).set_multimeter(connections, **kwargs)

    def set_spike_stimulus(self, connections=OrderedDict({}), **kwargs):
        kwargs["weights"] = kwargs.pop("weights", 1.0)
        kwargs["delays"] = kwargs.pop("delays", 0.0)
        kwargs["receptor_types"] = kwargs.pop("receptor_types", lambda target_node: target_node + 1)
        return super(WWDeco2014Builder, self).set_spike_stimulus(connections, **kwargs)

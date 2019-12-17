# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.builders.base import NESTModelBuilder


class WWDeco2014Builder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
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
                 ):
        config.nest.DEFAULT_MODEL = "iaf_cond_deco2014"
        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Common order of neurons' number per population:
        self.population_order = 100

        # Populations' configurations
        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        exc_pop_scale = 1.0
        exc_pop_size = int(self.population_order * exc_pop_scale)

        common_params = {
            "V_th": V_th, "V_reset": V_reset, "E_L": E_L,  "E_ex": E_ex,  "E_in": E_in,
            "tau_decay_AMPA": tau_decay_AMPA, "tau_decay_GABA_A": tau_decay_GABA,
            "tau_decay_NMDA": tau_decay_NMDA, "tau_rise_NMDA": tau_rise_NMDA,
        }
        self.params_ex = dict(common_params)
        self.params_ex.update({
            "C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex,
            "g_AMPA_ext": g_AMPA_ext_ex, "g_AMPA": g_AMPA_rec_ex, "g_NMDA": g_NMDA_ex, "g_GABA_A": g_GABA_ex,
            "w_E": self.tvb_model.w_p[0], "w_I": self.tvb_model.J_i[0], "N_E": exc_pop_size
        })
        self.params_in = dict(common_params)
        self.params_in.update({
            "C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in,
            "g_AMPA_ext": g_AMPA_ext_in, "g_AMPA": g_AMPA_rec_in, "g_NMDA": g_NMDA_in, "g_GABA_A": g_GABA_in,
            "w_E": 1.0, "w_I": 1.0, "N_E": exc_pop_size
        })

        def param_fun(node_index, params):
            out_params = dict(params)
            out_params.update({"w_E_ext": 100 * self.tvb_model.G[0] *
                                                     self.tvb_weights[:, list(self.nest_nodes_ids).index(node_index)]})
            return out_params

        self.populations = [{"label": "E", "model": self.default_population["model"],
                             "nodes": None,  # None means "all"
                             "params": lambda node_index: param_fun(node_index, self.params_ex),
                             "scale": exc_pop_scale},
                            {"label": "I", "model": self.default_population["model"],
                             "nodes": None,  # None means "all"
                             "params": lambda node_index: param_fun(node_index, self.params_in),
                             "scale": 1.0}
                           ]

        # Within region-node connections
        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations_connections = [
            {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": 1.0,
             "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "E", "target": "I",  # E -> I
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": 1.0,
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "E",  # I -> E
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": -1.0,
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None},  # None means "all"
             {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": -1.0,
              "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None}  # None means "all"
                                    ]

        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # depends on regions, set a handle to a function with
        # arguments (source_region_index=None, target_region_index=None)

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.nodes_connections = [
            {"source": "E", "target": ["E", "I"],
             "model": self.default_nodes_connection["model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": 100.0,  # weight scaling the TVB connectivity weight
             "delay": self.default_nodes_connection["delay"],  # additional delay to the one of TVB connectivity
             # Each region emits spikes in its own port:
             "receptor_type": lambda source_region_index, target_region_index=None: int(source_region_index + 1),
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
                                    "connections": connections, "nodes": None})  # None means "all"
        connections = OrderedDict({})
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.nest_instance.GetKernelStatus("resolution")  # self.monitor_period
        params['record_from'] = ["V_m",
                                 "s_AMPA", "x_NMDA", "s_NMDA", "s_GABA",
                                 "I_AMPA", "I_NMDA", "I_GABA", "I_L", "I_e",
                                 "spikes_exc", "spikes_inh"
                                 ]
        for i_node in range(self.number_of_nodes):
            params['record_from'].append("s_AMPA_ext_%d" % i_node)
            params['record_from'].append("I_AMPA_ext_%d" % i_node)
            params['record_from'].append("spikes_exc_ext_%d" % i_node)
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means "all"


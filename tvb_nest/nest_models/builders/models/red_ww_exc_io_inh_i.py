# -*- coding: utf-8 -*-

from collections import OrderedDict

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class RedWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
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
                 tau_rise_in=1.0  # assuming tau_rise_GABA = 1.0 ms
                 ):
        # config.DEFAULT_MODEL = "iaf_cond_deco2014"
        super(RedWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Common order of neurons' number per population:
        self.population_order = 100

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
        self.populations = [{"label": "E", "model": self.default_population["model"],
                             "nodes": None,  # None means "all"
                             "params": self.params_ex,
                             "scale": 1.6},
                            {"label": "I", "model": self.default_population["model"],
                             "nodes": None,  # None means "all"
                             "params": self.params_in,
                             "scale": 0.4}
                            ]

        # Within region-node connections
        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations_connections = [
            {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.tvb_model.w_p[0],
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
             "weight": -self.tvb_model.J_i[0],
             "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -1.0,
             "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None}  # None means "all"
        ]
        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.nodes_connections = [
            {"source": "E", "target": ["E"],
             "model": self.default_nodes_connection["model"],
             "conn_spec": self.default_nodes_connection["conn_spec"],
             "weight": self.G_scale_tvb_weight_exc,  # weight scaling the TVB connectivity weight
             "delay": self.tvb_delay,
             # Each region emits spikes in its own port:
             "receptor_type": 0,
             "source_nodes": None, "target_nodes": None}  # None means "all"
                                 ]
        if self.tvb_model.lamda[0] > 0:
            self.nodes_connections.append(
                {"source": "E", "target": ["I"],
                 "model": self.default_nodes_connection["model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": self.G_scale_tvb_weight_inh,  # weight scaling the TVB connectivity weight
                 "delay": self.tvb_delay,  # additional delay to the one of TVB connectivity
                 # Each region emits spikes in its own port:
                 "receptor_type": 0,
                 "source_nodes": None, "target_nodes": None}
            )

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
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means all here

    # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!

    def G_scale_tvb_weight_exc(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.tvb_model.G[0].item() * self.tvb_simulator.coupling.a[0].item())

    def G_scale_tvb_weight_inh(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, self.tvb_weights,
                                scale=self.tvb_model.lamda[0].item() *
                                      self.tvb_model.G[0].item() * self.tvb_simulator.coupling.a[0].item())

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

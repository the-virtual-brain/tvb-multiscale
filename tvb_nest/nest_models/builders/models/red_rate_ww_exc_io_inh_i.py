# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class RedRateWWExcIOInhIBuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 J_N=150, J_i=1000):
        config.DEFAULT_CONNECTION["model"] = "rate_connection"
        super(RedRateWWExcIOInhIBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.J_N = J_N
        self.J_i = J_i

        # Compile if builders are missing
        # Here there is one module for both builders,
        # so that the default naming pattern would work...:
        self._confirm_compile_install_nest_models(["tvb_rate_redwongwang_exc",
                                                   "tvb_rate_redwongwang_inh"], modules="tvb_rate_wongwangmodule")

        # Common order of neurons' number per population:
        self.population_order = 100
        # Populations' configurations
        self.populations = [{"label": "E", "model": "tvb_rate_redwongwang_exc",
                             "scale": 1, "params": {}, "nodes": None},  # None means "all"
                            {"label": "I", "model": "tvb_rate_redwongwang_inh",
                             "scale": 0.7, "params": {}, "nodes": None}]  # None means "all"

        # Within region-node connections
        # Connection weights between the distinct populations:
        # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
        self.populations_connections = [
            {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.J_N, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "E", "target": "I",  # E -> I
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.J_N, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "E",  # I -> E
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": 0, "nodes": None},  # None means "all"
        ]
        # Among/Between region-node connections
        # Given that w_ee == w_ie = J_N,
        # and that only the excitatory population of one region-node couples to
        # both excitatory and inhibitory populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"source": "E", "target": ["E"],
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": self.J_N_G_scale_tvb_weight_exc,
              "delay": self.tvb_delay,
              "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
             ]
        if self.tvb_model.lamda[0] > 0:
            self.node_connections.append(
                {"source": "E", "target": ["I"],
                 "model": self.default_nodes_connection["model"],
                 "conn_spec": self.default_nodes_connection["conn_spec"],
                 "weight": self.J_N_G_scale_tvb_weight_inh,
                 "delay": self.tvb_delay,
                 "receptor_type": 0, "source_nodes": None, "target_nodes": None}
            )
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        connections = OrderedDict({})
        #          label <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        params['record_from'] = ["S", "rate", "I_syn", "currents"]
        self.output_devices = [{"model": "multimeter", "params": params,
                                "connections": connections, "nodes": None}]  # None means "all"

    def J_N_G_scale_tvb_weight_exc(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                scale=self.J_N[0] * self.G[0])

    def J_N_G_scale_tvb_weight_inh(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                scale=self.J_N[0] * self.lamda[0] * self.G[0])

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

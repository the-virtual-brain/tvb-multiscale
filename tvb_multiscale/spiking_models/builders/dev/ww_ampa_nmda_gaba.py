# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class WWAMPANMDAGABABuilder(SpikingModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 w_ee=1.4, J_i=1.0):
        # Some properties for the default synapse to be used:
        config.DEFAULT_CONNECTION["model"] = "scale_connection"
        super(WWAMPANMDAGABABuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.w_ee = w_ee
        self.J_i = J_i

        self.populations_params = [{}, {},  # AMPA and NMDA get the default parameters
                                   {"tau_syn": 10.0}]  # decay synaptic time for GABA has to change

        # Common order of neurons' number per population:
        self.population_order = 100
        # Populations' configurations
        self.populations = [{"label": "AMPA", "model": "iaf_cond_ampa_gaba_deco2014",
                             "scale": 1, "params": {}, "nodes": None},  # None means "all"
                            {"label": "NMDA", "model": "iaf_cond_nmda_deco2014",
                             "scale": 1, "params": {}, "nodes": None},  # None means "all"
                            {"label": "I", "model": "iaf_cond_ampa_gaba_deco2014",
                             "scale": 0.7, "params": {"tau_syn": 10.0}, "nodes": None}]  # None means "all"

        # Spiking populations scalings for the number of neurons:
        rcptr_ampa_gaba = self.nest_instance.GetDefaults('iaf_cond_ampa_gaba_deco2014')['receptor_types']
        rcptr_nmda = self.nest_instance.GetDefaults('iaf_cond_nmda_deco2014')['receptor_types']

        # Within region-node connections
        # Connection weights between the distinct populations:
        # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
        self.populations_connections = [
            {"source": "AMPA", "target": "AMPA",  # AMPA -> AMPA This is a self-connection for population "AMPA"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESEXC_AMPA_REC"], "nodes": None},  # None means "all"
            {"source": "NMDA", "target": "AMPA",  # NMDA -> AMPA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESEXC_NMDA"], "nodes": None},  # None means "all"
            {"source": "GABA", "target": "AMPA",  # GABA -> AMPA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESINH_GABA"], "nodes": None},  # None means "all"

            {"source": "AMPA", "target": "NMDA",  # AMPA -> NMDA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_nmda["SPIKESEXC_AMPA_REC"], "nodes": None},  # None means "all"
            {"source": "NMDA", "target": "NMDA",  # NMDA -> NMDA  This is a self-connection for population "NMDA"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": self.w_ee, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_nmda["SPIKESEXC_NMDA"], "nodes": None},  # None means "all"
            {"source": "GABA", "target": "NMDA",  # GABA -> NMDA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -self.J_i, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_nmda["SPIKESINH_GABA"], "nodes": None},  # None means "all"

            {"source": "AMPA", "target": "GABA",  # AMPA -> GABA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": 1.0, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESEXC_AMPA_REC"], "nodes": None},  # None means "all"
            {"source": "NMDA", "target": "GABA",  # NMDA -> GABA
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": 1.0, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESEXC_NMDA"], "nodes": None},  # None means "all"
            {"source": "GABA", "target": "GABA",  # GABA -> GABA This is a self-connection for population "GABA"
             "model": self.default_populations_connection["model"],
             "conn_spec": self.default_populations_connection["conn_spec"],
             "weight": -1.0, "delay": self.default_populations_connection["delay"],
             "receptor_type": rcptr_ampa_gaba["SPIKESINH_GABA"], "nodes": None},  # None means "all"
        ]

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"source": "AMPA", "target": ["AMPA", "NMDA", "GABA"],
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": self.G_scale_tvb_weight,  # weight scaling the TVB connectivity weight
              "delay": tvb_delay,  # additional delay to the one of TVB connectivity
              "receptor_type": rcptr_ampa_gaba["SPIKESEXC_AMPA_EXT"]},
             ]

        # Creating  devices to be able to observe NEST activity:
        self.output_devices = []
        connections = OrderedDict({})
        #                   label <- target population
        connections["AMPA spikes"] = "AMPA"
        connections["NMDA spikes"] = "NMDA"
        connections["GABA spikes"] = "GABA"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"])
        self.output_devices.append({"model": "spike_detector", "params": params,
                                    "connections": connections, "nodes": None})  # None means "all"
        connections = OrderedDict({})
        #               label <- target population
        connections["AMPA"] = "AMPA"
        connections["NMDA"] = "NMDA"
        connections["GABA"] = "GABA"
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.monitor_period
        params['record_from'] = ["V_m", "s", "s_bound", "emmited_spike",
                                 "s_AMPA_ext", "s_AMPA_rec", "s_NMDA", "s_GABA",
                                 "I_AMPA_ext", "I_AMPA_rec", "I_NMDA", "I_GABA", "I_leak"]
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "connections": connections, "nodes": None})  # None means "all"

    def G_scale_tvb_weight(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node, scale=self.tvb_model.G[0])
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.tvb_netpyne.config import CONFIGURED
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import \
    random_normal_weight, random_normal_tvb_weight, random_uniform_tvb_delay


class DefaultExcIOInhIBuilder(NetpyneNetworkBuilder):

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                      spiking_simulator=spiking_simulator, config=config, logger=logger)

        self.scale_e = 1.2
        self.scale_i = 0.4

        self.w_ee = 0.02
        self.w_ei = 0.01
        self.w_ie = 0.01
        self.w_ii = 0.01

        self.delay = 5

        self.conn_spec_all_to_all = {"rule": "all_to_all"}
        self.conn_spec_prob_low = {"rule": {"prob": 0.01}}
        self.conn_spec_prob_high = {"rule": {"prob": 0.03}}

        self.params = {} # TODO: pass some info at least about transforming the default delay (because TVB support only delay as single number)

    def configure(self):
        from netpyne import specs
        netParams = specs.NetParams()
        cfg = specs.SimConfig()

        self.receptor_type_E = 'exc'
        self.receptor_type_I = 'inh'
        netParams.synMechParams[self.receptor_type_E] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA
        netParams.synMechParams[self.receptor_type_I] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA

        PYRcell = {'secs': dict()}
        PYRcell['secs']['soma'] = {'geom': {}, 'mechs': {}}  # soma params dict
        PYRcell['secs']['soma']['geom'] = {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}  # soma geometry
        PYRcell['secs']['soma']['mechs']['hh'] = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
        netParams.cellParams['PYR'] = PYRcell

        # simConfig.verbose = True

        cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        
        cfg.recordStep = 0.1
        cfg.savePickle = False        # Save params, network and sim output to pickle file
        cfg.saveJson = False

        super(DefaultExcIOInhIBuilder, self).configure(netParams, cfg, autoCreateSpikingNodes=True)
        self.global_coupling_scaling *= self.tvb_serial_sim.get("model.G", np.array([2.0]))[0].item()
        self.lamda = self.tvb_serial_sim.get("model.lamda", np.array([0.0]))[0].item()

    def proxy_node_synaptic_model_funcs(self):
        return {"E": lambda src_node, dst_node: self.receptor_type_E,
                "I": lambda src_node, dst_node: self.receptor_type_E}

    def set_defaults(self):
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        # self.set_input_devices()


    def set_populations(self):
        self.populations = [
            {"label": "E", "model": "PYR",
             "nodes": None,  # None means "all"
             "params": self.params,
             "scale": self.scale_e},
            {"label": "I", "model": "PYR",
             "nodes": None,  # None means "all"
             "params": self.params,
        #  "params": netpyne_network_builder.params_I,
             "scale": self.scale_i}]

    # connections between E and I populations of same spiking node

    def set_populations_connections(self):
        self.populations_connections = [
            self.set_EE_populations_connections(),
            self.set_EI_populations_connections(),
            self.set_IE_populations_connections(),
            self.set_II_populations_connections()
        ]

    def set_EE_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",
            #  "synapse_model": self.default_populations_connection["synapse_model"],
             # TODO: here and below, is this needed or `receptor_type` below is just enough?
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.weight_fun_ee,
             "delay": self.delay,
             "receptor_type": self.receptor_type_E, "nodes": None}  # None means "all"
        # connections.update(self.pop_conns_EE)
        return connections

    def set_EI_populations_connections(self):
        connections = \
            {"source": "E", "target": "I",
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.weight_fun_ei,
             "delay": self.delay,
             "receptor_type": self.receptor_type_E, "nodes": None}  # None means "all"
        return connections

    def set_IE_populations_connections(self):
        connections = \
            {"source": "I", "target": "E",
             "conn_spec": self.conn_spec_prob_high,
             "weight": self.weight_fun_ie,
             "delay": self.delay,
             "receptor_type": self.receptor_type_I, "nodes": None}  # None means "all"
        return connections

    def set_II_populations_connections(self):
        connections = \
            {"source": "I", "target": "I",
             "conn_spec": self.conn_spec_prob_high,
             "weight": self.weight_fun_ii,
             "delay": self.delay,
             "receptor_type": self.receptor_type_I, "nodes": None}  # None means "all"
        return connections

    def weight_fun_ee(self, nodeId):
        return self.within_node_weight_fun(self.w_ee, nodeId)

    def weight_fun_ei(self, nodeId):
        return self.within_node_weight_fun(self.w_ei, nodeId)

    def weight_fun_ie(self, nodeId):
        return self.within_node_weight_fun(self.w_ie, nodeId)

    def weight_fun_ii(self, nodeId):
        return self.within_node_weight_fun(self.w_ii, nodeId)

    def within_node_weight_fun(self, weight, nodeId):
        weights = ensure_list(weight)
        weight = weights[nodeId] if (len(weights) > 1) else weights[0]
        return random_normal_weight(weight)

    # connections between population of different spiking nodes

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": "E",
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": self.receptor_type_E, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        if self.lamda > 0:
            self.nodes_connections.append({
                "source": "E", "target": "I",
                "conn_spec": self.conn_spec_prob_low,
                # using lamda to scale connectivity weights (or alternatively,
                # it can be used to downscale connection probability in 'conn_spec' above):
                "weight": lambda source_node, target_node: self.tvb_weight_fun(source_node, target_node, self.lamda),
                "delay": self.tvb_delay_fun,
                # Each region emits spikes in its own port:
                "receptor_type": self.receptor_type_E, "source_nodes": None, "target_nodes": None  # None means "all"
            })

    # output devices

    def set_output_devices(self):
        # Creating  devices to be able to observe NetPyNE activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_recorder()]

    def set_spike_recorder(self):
        connections = OrderedDict()
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        params = self.config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
        # params["record_to"] = self.output_devices_record_to
        device = {"model": "spike_recorder", "params": params,
                #   "neurons_fun": lambda node_id, population: population[:np.minimum(100, len(population))],
                  "connections": connections, "nodes": None}  # None means all here
        # device.update(self.spike_recorder)
        return device

    def tvb_weight_fun(self, source_node, target_node, lamda=None, sigma=0.1):
        scale = self.global_coupling_scaling * self.netpyne_synaptic_weight_scale
        if lamda:
            scale *= lamda
        return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale=scale, sigma=sigma)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        if low is None:
            low = self.tvb_dt
        if high is None:
            high = 2*low
        return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(DefaultExcIOInhIBuilder, self).build()

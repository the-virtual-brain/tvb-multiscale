from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
from tvb_multiscale.tvb_netpyne.netpyne.data_structures import NetpyneCellGeometry, NetpyneCellModel, NetpyneMechanism
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_weight, random_normal_tvb_weight, random_uniform_tvb_delay
from collections import OrderedDict
import numpy as np

class DefaultExcIOInhIBuilder(NetpyneNetworkBuilder):

    def __init__(self, tvb_simulator={}, spiking_nodes_inds=[], netpyne_instance=None, config=None):
        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, netpyne_instance=netpyne_instance, config=config)

        self.scale_e = 1.2
        self.scale_i = 0.4

        self.weight_ee = 0.02
        self.weight_ei = 0.01
        self.weight_ie = 0.01
        self.weight_ii = 0.01

        self.delay = 5

        self.conn_spec_all_to_all = {"rule": "all_to_all"}
        self.conn_spec_prob_low = {"rule": {"prob": 0.01}}
        self.conn_spec_prob_high = {"rule": {"prob": 0.03}}

        self.params = {} # TODO: pass some info at least about transforming the default delay (because TVB support only delay as single number)

    def configure(self):
        super(DefaultExcIOInhIBuilder, self).configure()
        self.global_coupling_scaling *= self.tvb_serial_sim.get("model.G", np.array([2.0]))[0].item()

    def set_defaults(self):
        self.set_cell_models()
        self.set_synapse_models()
        self.set_populations()
        self.set_populations_connections()
        self.set_nodes_connections()
        self.set_output_devices()
        # self.set_input_devices()

    def set_cell_models(self):
        geom = NetpyneCellGeometry(diam=18.8, length=18.8, axialResistance=123)
        mech = NetpyneMechanism(name='hh', gNaBar=0.12, gKBar=0.036, gLeak=0.003, eLeak=-70)
        cell_model = NetpyneCellModel(name='PYR', geom=geom, mech=mech)
        self.cell_models = [cell_model]

        self.cell_model_E = cell_model
        self.cell_model_I = cell_model

    def set_synapse_models(self):
        # TODO: move all params here from NetpyneInstance
        self.receptor_type_E = "exc"
        self.receptor_type_I = "inh"

    def set_populations(self):
        self.populations = [
            {"label": "E", "model": self.cell_model_E.name,
             "nodes": None,  # None means "all"
             "params": self.params,
             "scale": self.scale_e},
            {"label": "I", "model": self.cell_model_I.name,
             "nodes": None,  # None means "all"
             "params": self.params,
        #  "params": netpyne_network_builder.params_I,
             "scale": self.scale_i}]

    def set_populations_connections(self):
        self.populations_connections = [
            self.set_EE_populations_connections(),
            self.set_EI_populations_connections(),
            self.set_IE_populations_connections(),
            self.set_II_populations_connections()
        ]

    # def receptor_E_fun(self):
    #     return 0

    # def receptor_I_fun(self):
    #     return 0

    def set_EE_populations_connections(self):
        connections = \
            {"source": "E", "target": "E",
            #  "synapse_model": self.default_populations_connection["synapse_model"], # TODO: here and below, is this needed or `receptor_type` below is just enough?
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.weight_fun(self.weight_ee),
             "delay": self.delay,
             "receptor_type": self.receptor_type_E, "nodes": None}  # None means "all"
        # connections.update(self.pop_conns_EE)
        return connections

    def set_EI_populations_connections(self):
        connections = \
            {"source": "E", "target": "I",
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.weight_fun(self.weight_ei),
             "delay": self.delay,
             "receptor_type": self.receptor_type_E, "nodes": None}  # None means "all"
        return connections

    def set_IE_populations_connections(self):
        connections = \
            {"source": "I", "target": "E",
             "conn_spec": self.conn_spec_prob_high,
             "weight": self.weight_fun(self.weight_ie),
             "delay": self.delay,
             "receptor_type": self.receptor_type_I, "nodes": None}  # None means "all"
        return connections

    def set_II_populations_connections(self):
        connections = \
            {"source": "I", "target": "I",
             "conn_spec": self.conn_spec_prob_high,
             "weight": self.weight_fun(self.weight_ii),
             "delay": self.delay,
             "receptor_type": self.receptor_type_I, "nodes": None}  # None means "all"
        return connections

    def set_nodes_connections(self):
        self.nodes_connections = [
            {"source": "E", "target": "E",
             "conn_spec": self.conn_spec_prob_low,
             "weight": self.tvb_weight_fun,
             "delay": self.tvb_delay_fun,
             # Each region emits spikes in its own port:
             "receptor_type": self.receptor_type_E, "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
        # TODO: lamda?

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

    def weight_fun(self, w, scale=1.0, sigma=0.1):
        return random_normal_weight(w, scale, sigma)

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.netpyne_synaptic_weight_scale * self.global_coupling_scaling #TODO: this is not proper scaling, need to by multiplied
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
        self.configureCells()
        return super(DefaultExcIOInhIBuilder, self).build()
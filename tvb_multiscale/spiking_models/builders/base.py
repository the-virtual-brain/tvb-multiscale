# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np
from pandas import Series
from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.spiking_models.network import SpikingNetwork
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list, flatten_tuple, property_to_fun


LOG = initialize_logger(__name__)


class SpikingModelBuilder(object):
    __metaclass__ = ABCMeta

    config = CONFIGURED
    default_synaptic_weight_scaling = \
        lambda self, weight, n_cons: self.config.DEFAULT_SPIKING_SYNAPTIC_WEIGHT_SCALING(weight, n_cons)
    nodes = []
    monitor_period = 1.0
    tvb_to_spiking_dt_ratio = 10
    spiking_dt = 0.1 / tvb_to_spiking_dt_ratio
    spiking_min_delay_ratio = 4
    spiking_min_delay = 0.001
    spiking_nodes_ids = []
    _spiking_nodes_labels = []

    population_order = 100

    default_population = {}
    default_populations_connection = {}
    default_nodes_connection = {}

    populations = []
    _populations = []
    populations_connections = []
    _populations_connections = []
    nodes_connections = []
    _nodes_connections = []
    # Use these to observe Spiking Simnulator behavior without conversions to TVB state variables and monitors
    output_devices = []
    _output_devices = []
    input_devices = []  # use these for possible external stimulation devices
    _input_devices = []
    # Example:
    # input_devices = [{"model": "poisson_generator",
    # ---------Properties potentially set as function handles with args (spiking_node_id=None)-----------------------------
    #                         "params": {"rate": 50.0,
    #                                    "origin": 10.0,
    #                                    "start": 0.1,
    #                                    "stop": 20.0},
    #                         "weights": 1.0, "delays": 0.0
    #                         "connections: {"spikes": "E"},
    # ---------Properties potentially set as function handles with args (spiking_node_id=None)-----------------------------
    #                         "nodes": None}]  # None means "all" here

    def __init__(self, tvb_simulator, spiking_nodes_ids, config=CONFIGURED, logger=LOG):
        self.config = config
        self.logger = logger
        self.spiking_nodes_ids = np.unique(spiking_nodes_ids)
        self.tvb_simulator = tvb_simulator
        self.tvb_weights = self.tvb_connectivity.scaled_weights(mode='region')
        self.tvb_to_spiking_dt_ratio = 10
        self._update_spiking_dt()
        self.spiking_min_delay_ratio = 4
        self._update_spiking_min_delay()
        # We assume that there at least the Raw monitor which is also used for communication to/from Spiking Simulator
        # If there is only the Raw monitor, then self.monitor_period = self.tvb_dt
        self.monitor_period = tvb_simulator.monitors[-1].period
        self.population_order = 100

    @abstractmethod
    def build_spiking_populations(self, model, size, params, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def _spiking_min_delay(self):
        pass

    @abstractmethod
    def _prepare_populations_connection_params(self, pop_src, pop_trg, conn_spec, syn_spec):
        pass

    @abstractmethod
    def connect_two_populations(self, source, target, conn_params, synapse_params):
        pass

    @abstractmethod
    def build_and_connect_devices(self, devices):
        pass

    @property
    def tvb_model(self):
        return self.tvb_simulator.model

    @property
    def tvb_connectivity(self):
        return self.tvb_simulator.connectivity

    @property
    def tvb_delays(self):
        return self.tvb_connectivity.delays

    @property
    def tvb_dt(self):
        return self.tvb_simulator.integrator.dt

    @property
    def number_of_nodes(self):
        return self.tvb_connectivity.number_of_regions

    @property
    def number_of_spiking_nodes(self):
        return np.maximum(len(self.spiking_nodes_ids), 1)

    @property
    def spiking_nodes_labels(self):
        if len(self._spiking_nodes_labels) == self.number_of_spiking_nodes:
            return self._spiking_nodes_labels
        else:
            return self.tvb_connectivity.region_labels[self.spiking_nodes_ids.tolist()]

    # def _population_property_per_node(self, property):
    #     output = OrderedDict()
    #     spiking_nodes_labels = self.spiking_nodes_labels
    #     for population in self.populations:
    #         output[population["label"]] = property_per_node(population[property],
    #                                                         population.get("nodes", self.nest_nodes_ids),
    #                                                         self.nest_nodes_ids, spiking_nodes_labels)
    #     return output

    @property
    def number_of_populations(self):
        return len(self.populations)

    # @property
    # def populations_models(self):
    #     return self._population_property_per_node("model")
    #
    # @property
    # def populations_nodes(self):
    #     return self._population_property_per_node("nodes")
    #
    # @property
    # def populations_scales(self):
    #     return self._population_property_per_node("scales")
    #
    # @property
    # def populations_sizes(self):
    #     scales = self._population_property_per_node("scales")
    #     for pop_name, scale in scales.items():
    #         if isinstance(scale, dict):
    #             for node_key, node_scale in scale.items():
    #                 scales[pop_name][node_key] = int(np.round(scales[pop_name][node_key] * self.population_order))
    #         else:
    #             scales[pop_name] *= self.population_order
    #     return scales
    #
    # @property
    # def populations_params(self):
    #     return self._population_property_per_node("params")
    #
    # def _connection_label(self, connection):
    #     return "%s<-%s" % (str(connection["source"]), str(connection["target"]))
    #
    # def _connection_property_per_node(self, property, connections):
    #     spiking_nodes_labels = self.spiking_nodes_labels
    #     output = OrderedDict()
    #     for conn in connections:
    #         output[self._connection_label(conn)] = self._property_per_node(conn[property],
    #                                                                        conn.get("nodes", self.nest_nodes_ids),
    #                                                                        self.nest_nodes_ids, spiking_nodes_labels)
    #     return output
    #
    # def _population_connection_property_per_node(self, property):
    #     return self._connection_property_per_node(property, self.populations_connections)
    #
    # @property
    # def populations_connections_labels(self):
    #     return [self._connection_label(conn) for conn in self.populations_connections]
    #
    # @property
    # def populations_connections_models(self):
    #     return self._population_connection_property_per_node("model")
    #
    # @property
    # def populations_connections_weights(self):
    #     return self._population_connection_property_per_node("weight")
    #
    # @property
    # def populations_connections_delays(self):
    #     return self._population_connection_property_per_node("delay")
    #
    # @property
    # def populations_connections_receptor_types(self):
    #     return self._population_connection_property_per_node("receptor_type")
    #
    # @property
    # def populations_connections_conn_spec(self):
    #     return self._population_connection_property_per_node("conn_spec")
    #
    # @property
    # def populations_connections_nodes(self):
    #     return self._population_connection_property_per_node("nodes")
    #
    # def _nodes_connection_property_per_node(self, property, connections):
    #     spiking_nodes_labels = self.spiking_nodes_labels
    #     output = OrderedDict()
    #     for conn in connections:
    #         output[self._connection_label(conn)] = \
    #             property_per_nodes_connection(conn[property],
    #                                           conn.get("source_nodes", self.nest_nodes_ids),
    #                                           conn.get("target_nodes", self.nest_nodes_ids),
    #                                           self.nest_nodes_ids, spiking_nodes_labels)
    #     return output
    #
    # @property
    # def nodes_connections_labels(self):
    #     return [self._connection_label(conn) for conn in self.nodes_connections]
    #
    # @property
    # def nodes_connections_models(self):
    #     return self._nodes_connection_property_per_node("model")
    #
    # @property
    # def nodes_connections_weights(self):
    #     return self._nodes_connection_property_per_node("weight")
    #
    # @property
    # def nodes_connections_delays(self):
    #     return self._nodes_connection_property_per_node("delay")
    #
    # @property
    # def nodes_connections_receptor_types(self):
    #     return self._nodes_connection_property_per_node("receptor_type")
    #
    # @property
    # def nodes_connections_receptor_conn_spec(self):
    #     return self._nodes_connection_property_per_node("conn_spec")
    #
    # @property
    # def nodes_connections_source_nodes(self):
    #     return self._population_connection_property_per_node("source_nodes")
    #
    # @property
    # def nodes_connections_target_nodes(self):
    #     return self._population_connection_property_per_node("target_nodes")

    def _assert_within_node_delay(self, delay):
        if delay > self.tvb_dt / 2:
            if delay > self.tvb_dt:
                raise ValueError("Within Spiking nodes delay %f is not smaller "
                                 "than the TVB integration time step %f!"
                                 % (delay, self.tvb_dt))
            else:
                LOG.warning("Within Spiking nodes delay %f is not smaller "
                            "than half the TVB integration time step %f!"
                            % (delay, self.tvb_dt))
        return self._assert_delay(delay)

    def _update_spiking_dt(self):
        self.spiking_dt = \
            float(int(np.round(self.tvb_dt / self.tvb_to_spiking_dt_ratio / self._spiking_min_delay))) \
            * self._spiking_min_delay

    def _update_spiking_min_delay(self):
        self.spiking_min_delay = np.minimum(self.spiking_min_delay_ratio * self.spiking_dt, self.tvb_dt/2)

    def _configure_populations(self):
        # Every population must have his own model model,
        # scale of spiking neurons' number, and model specific parameters
        self.models = []
        self.populations_labels = []
        for i_pop, population in enumerate(self.populations):
            self._populations.append(dict(self.default_population))
            self._populations[-1].update(population)
            if len(self._populations[-1].get("label", "")) == 0:
                self._populations[-1]["label"] = "Pop%d" % i_pop
            self.populations_labels.append(self._populations[-1]["label"])
            if self._populations[-1]["nodes"] is None:
                self._populations[-1]["nodes"] = self.spiking_nodes_ids
            self.models.append(self._populations[-1]["model"])
            self._populations[-1]["scale"] = property_to_fun(self._populations[-1]["scale"])
            self._populations[-1]["params"] = property_to_fun(self._populations[-1]["params"])
        self.populations_labels = np.unique(self.populations_labels).tolist()
        self.models = np.unique(self.models).tolist()

    def _assert_connection_populations(self, connection):
        for pop in ["source", "target"]:
            pops_labels = connection.get(pop, None)
        if pops_labels is None:
            raise_value_error("No %s population in connection!:\n%s" % (pop, str(connection)))
        for pop_lbl in ensure_list(pops_labels):
            assert pop_lbl in self.populations_labels
        return pops_labels

    def _configure_connections(self, connections, default_connection):
        out_connections = list(connections)
        for i_con, connection in enumerate(out_connections):
            self._assert_connection_populations(connection)
            temp_conn = dict(default_connection)
            temp_conn.update(connection)
            out_connections[i_con] = temp_conn
            for prop in ["weight", "delay", "receptor_type"]:
                out_connections[i_con][prop] = property_to_fun(connection[i_con][prop])
        return out_connections

    def _configure_populations_connections(self):
        self._populations_connections = self._configure_connections(self.populations_connections,
                                                                    self.default_populations_connection)
        for i_conn, connections in enumerate(self.populations_connections):
            if connections["nodes"] is None:
                self._populations_connections[i_conn]["nodes"] = self.spiking_nodes_ids

    def _configure_nodes_connections(self):
        self._nodes_connections = self._configure_connections(self.nodes_connections,
                                                              self.default_nodes_connection)
        for i_conn, connections in enumerate(self.nodes_connections):
            for pop in ["source", "target"]:
                this_pop = "%s_nodes" % pop
                if connections[this_pop] is None:
                    self._nodes_connections[i_conn][this_pop] = self.spiking_nodes_ids
        self.tvb_connectivity.configure()

    def _configure_devices(self, devices):
        # Configure devices by the variable model they measure or stimulate (Series),
        # population (Series),
        # and target node (Series) for faster reading
        _devices = list()
        for device in devices:
            _devices.append(dict(device))
            spiking_nodes = device.get("nodes", self.spiking_nodes_ids)
            if spiking_nodes is None:
                spiking_nodes = self.spiking_nodes_ids
            _devices[-1]["nodes"] = spiking_nodes
            # User inputs
            params = device.get("params", {})
            # ..converted to functions
            weights_fun = property_to_fun(device.get("weights", 1.0))
            delays_fun = property_to_fun(device.get("delays", 0.0))
            receptor_types_fun = property_to_fun(device.get("receptor_types", 0))
            # Defaults in arrays:
            shape = (len(devices["nodes"]),)
            receptor_types = np.zeros(shape).astype("i")
            params = np.tile(params, shape)
            # weights and delays might be dictionaries for distributions:
            weights = np.tile([1.0, shape]).astype("object")
            delays = np.tile([0.0], shape).astype("object")
            target_spiking_nodes_ids = \
                [np.where(self.spiking_nodes_ids == trg_node)[0][0] for trg_node in devices["nodes"]]
            for trg_node, i_trg in zip(devices["nodes"], target_spiking_nodes_ids):
                weights[i_trg] = weights_fun(trg_node)  # a function also fo self.tvb_weights
                delays[i_trg] = delays_fun(trg_node)    # a function also fo self.tvb_delays
                receptor_types[i_trg] = receptor_types_fun(trg_node)
            _devices["nodes"] = target_spiking_nodes_ids
            _devices["weights"] = weights
            _devices["delays"] = delays
            _devices["receptor_types"] = receptor_types
            _devices["params"] = params
        return _devices

    def _configure_output_devices(self):
        self._output_devices = self._configure_devices(self.output_devices)

    def _configure_input_devices(self):
        self._input_devices = self._configure_devices(self.input_devices)

    def configure(self):
        self._populations = self._configure_populations()
        self._populations_connections =self._configure_populations_connections()
        self._nodes_connections = self._configure_nodes_connections()
        self._configure_output_devices()
        self._configure_input_devices()

    def _synaptic_weight_scaling(self, weights, number_of_connections):
        return self.default_synaptic_weight_scaling(weights, number_of_connections)

    def _assert_delay(self, delay):
        assert delay >= 0.0
        return delay

    def build_spiking_nodes(self, *args, **kwargs):
        spiking_nodes_labels = self.spiking_nodes_labels
        self.nodes = Series()
        for node_id, node_label in zip(self.spiking_nodes_ids, spiking_nodes_labels):  # For every Spiking node
            for iP, population in enumerate(self._populations):
                if node_id in population["nodes"]:
                    # ...generate a network of spiking populations
                    size = population["scale"](node_id) * self.population_order
                    self.nodes[node_label][population["label"]] = \
                        self.build_spiking_populations(population["model"], size, params=population["params"],
                                                       *args, **kwargs)

    def _connect_two_populations(self, pop_src, pop_trg, conn_spec, syn_spec):
        conn_spec, n_cons = self._prepare_populations_connection_params(pop_src, pop_trg, conn_spec, syn_spec)
        # Scale the synaptic weight with respect to the total number of connections between the two populations:
        syn_spec["weight"] = self._synaptic_weight_scaling(syn_spec["weight"], n_cons)
        receptors = ensure_list(syn_spec["receptor_type"])
        for receptor in receptors:
            syn_spec["receptor_type"] = receptor
            self.connect_two_populations(pop_src, pop_trg, conn_spec, syn_spec)

    def _connect_two_populations_within_node(self, pop_src, pop_trg,
                                             conn_spec, syn_model, weight, delay, receptor_type):
        syn_spec = {'model': syn_model,
                    'weight': weight,
                    'delay': self._assert_within_node_delay(delay),
                    'receptor_type': receptor_type}
        self._connect_two_populations(pop_src, pop_trg, conn_spec, syn_spec)

    def _get_node_populations_neurons(self, node, populations):
        return flatten_tuple([node[pop] for pop in ensure_list(populations)])

    def connect_within_node_spiking_populations(self):
        # For every different type of connections between distinct Spiking nodes' populations
        for i_conn, conn in enumerate(ensure_list(self._populations_connections)):
            # ...and form the connection within each Spiking node
            for node_index in conn["nodes"]:
                i_node = np.where(self.spiking_nodes_ids == node_index)[0][0]
                self._connect_two_populations_within_node(
                    self._get_node_populations_neurons(self.nodes[i_node], conn["source"]),
                    self._get_node_populations_neurons(self.nodes[i_node], conn["target"]),
                    conn['conn_spec'], conn["model"],
                    conn['weight'](node_index), conn['delay'](node_index),
                    conn['receptor_type'](node_index))

    def _connect_two_populations_between_nodes(self, pop_src, pop_trg, i_n_src, i_n_trg,
                                               conn_spec, syn_model, weight, delay, receptor_type):
        src_node_id = self.spiking_nodes_ids[i_n_src]
        trg_node_id = self.spiking_nodes_ids[i_n_trg]
        syn_spec = {'model': syn_model,
                    'weight': self.tvb_weights[src_node_id, trg_node_id] * weight,
                    'delay': self.tvb_delays[src_node_id, trg_node_id] + delay,
                    'receptor_type': receptor_type}
        self._connect_two_populations(pop_src, pop_trg, conn_spec, syn_spec)

    def connect_spiking_nodes(self):
        # Define a function for the exact synthesis of source and target populations
        population = lambda node, populations: \
            flatten_tuple([node[pop] for pop in ensure_list(populations)])
        # For every different type of connections between distinct Spiking nodes' populations
        for i_conn, conn in enumerate(ensure_list(self._nodes_connections)):
            # ...and form the connection for every distinct pair of Spiking nodes
            for source_index in conn["source_nodes"]:
                i_source_node = np.where(self.spiking_nodes_ids == source_index)[0][0]
                for target_index in conn["target_nodes"]:
                    if source_index != target_index:  # TODO! Confirm that no self connections are allowed here!
                        i_target_node = np.where(self.spiking_nodes_ids == target_index)[0][0]
                        if self.tvb_weights[source_index, target_index] > 0:
                            self._connect_two_populations_between_nodes(population(self.nodes[i_source_node],
                                                                                   conn["source"]),
                                                                        population(self.nodes[i_target_node],
                                                                                   conn["target"]),
                                                                        i_source_node, i_target_node,
                                                                        conn['conn_spec'],
                                                                        conn["model"],
                                                                        conn["weight"](source_index, target_index),
                                                                        conn["delay"](source_index, target_index),
                                                                        conn["receptor_type"](source_index,
                                                                                              target_index))

    def _build_and_connect_devices(self, devices):
        # Build devices by the variable model they measure or stimulate (Series),
        # population (Series),
        # and target node (Series) for faster reading
        _devices = Series()
        for device in devices:
            _devices = _devices.append(
                            self.build_and_connect_devices(device))
        return _devices

    def build_and_connect_output_devices(self):
        # Build devices by the variable model they measure (Series),
        # population (Series),
        # and target node (Series) for faster reading
        return self._build_and_connect_devices(self._output_devices)

    def build_and_connect_input_devices(self):
        # Build devices by the variable model they stimulate (Series),
        # population (Series),
        # and target node (Series) for faster reading
        return self._build_and_connect_devices(self._input_devices)

    def build_spiking_network(self):
        # Configure all inputs to set them to the correct formats and sizes
        self.configure()
        # Build and connect internally all Spiking nodes
        self.build_spiking_nodes()
        self.connect_within_node_spiking_populations()
        # Connect Spiking nodes among each other
        self.connect_spiking_nodes()
        # Build and connect possible Spiking output devices
        # !!Use it only for extra Spiking quantities
        # that do not correspond to TVB state variables or parameters
        # you wish to transmit from Spiking to TVB!!
        output_devices = self.build_and_connect_output_devices()
        # Build and connect possible Spiking input devices
        # !!Use it only for stimuli, if any!!
        input_devices = self.build_and_connect_input_devices()
        return SpikingNetwork(self.nodes, output_devices, input_devices,
                              self.default_nodes_connection["delay"], self.config)


# def node_key_index_and_label(node, labels):
#     if isinstance(node, string_types):
#         try:
#             i_node = labels.index(node)
#             label = node
#             node_key = "%d-%s" % (node, i_node)
#         except:
#             raise_value_error("Node %s is not a region node modeled in Spiking Simulator!" % node)
#     else:
#         try:
#             label = labels.index(node)
#             i_node = node
#             node_key = "%d-%s" % (label, node)
#         except:
#             raise_value_error("Node %d is not a region node modeled in Spiking Simulator!" % node)
#     return node_key, i_node, label
#
#
# def property_per_node(property, nodes, spiking_nodes_labels):
#     if hasattr(property, "__call__"):
#         property_per_node = OrderedDict()
#         for node in nodes:
#             node_key, node_index = node_key_index_and_label(node, spiking_nodes_labels)[:2]
#             property_per_node[node_key] = property(node_index)
#         return property_per_node
#     else:
#         return property
#
#
# def property_per_nodes_connection(property, source_nodes, target_nodes, nest_nodes_ids, spiking_nodes_labels):
#     if hasattr(property, "__call__"):
#         if source_nodes is None:
#             source_nodes = nest_nodes_ids
#         else:
#             source_nodes = np.unique(source_nodes)
#         if target_nodes is None:
#             target_nodes = nest_nodes_ids
#         else:
#             target_nodes = np.unique(target_nodes)
#         property_per_nodes_connection = OrderedDict()
#         for source_node in source_nodes:
#             source_index, source_label = node_key_index_and_label(source_node, spiking_nodes_labels)[1:]
#             for target_node in target_nodes:
#                 target_index, target_label = node_key_index_and_label(target_node, spiking_nodes_labels)[1:]
#             node_connection_label = "%d.%s<-%d.%s" % (source_index, source_label, target_index, target_label)
#             property_per_nodes_connection[node_connection_label] = property(source_index, target_index)
#         return property_per_nodes_connection
#     else:
#         return property

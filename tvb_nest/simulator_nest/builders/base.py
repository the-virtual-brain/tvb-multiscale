# -*- coding: utf-8 -*-
from six import string_types
from collections import OrderedDict
from itertools import cycle
from pandas import Series
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models.region_node import NESTRegionNode
from tvb_nest.simulator_nest.models.network import NESTNetwork
from tvb_nest.simulator_nest.nest_factory import *
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list, flatten_tuple, property_to_fun


LOG = initialize_logger(__name__)


class NESTModelBuilder(object):
    config = CONFIGURED.nest
    default_synaptic_weight_scaling = \
        lambda self, weight, n_cons: self.config.DEFAULT_NEST_SYNAPTIC_WEIGHT_SCALING(weight, n_cons)
    nest_instance = None
    nodes = []
    monitor_period = 1.0
    tvb_to_nest_dt_ratio = 4
    nest_dt = 0.1 / tvb_to_nest_dt_ratio
    nest_nodes_ids = []
    _nest_nodes_labels = []

    population_order = 100
    populations = []
    populations_connections = []
    nodes_connections = []
    # Use these to observe NEST behavior without conversions to TVB state variables and monitors
    output_devices = []
    stimulation_devices = []  # use these for possible external stimulation devices

    # Example:
    # stimulation_devices = [{"model": "poisson_generator",
    # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
    #                         "params": {"rate": 50.0,
    #                                    "origin": 10.0,
    #                                    "start": 0.1,
    #                                    "stop": 20.0},
    #                         "weights": 1.0, "delays": 0.0
    #                         "connections: {"spikes": "E"},
    # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
    #                         "nodes": None}]  # None means "all" here

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, logger=LOG):
        self.config = config
        self.logger = logger
        # Setting or loading a nest instance:
        if nest_instance is not None:
            self.nest_instance = nest_instance
        else:
            self.nest_instance = load_spiking_simulator(self.config.nest, self.logger)

        # Setting NEST defaults from config
        self.default_population = {"model": self.config.nest.DEFAULT_MODEL, "scale": 1, "params": {}, "nodes": None}
        self.default_populations_connection = dict(self.config.nest.DEFAULT_CONNECTION)
        self.default_populations_connection["nodes"] = None
        self.default_nodes_connection = dict(self.config.nest.DEFAULT_CONNECTION)
        self.default_nodes_connection.update({"source_nodes": None, "target_nodes": None})
        self.default_synaptic_weight_scaling = \
            lambda weight, n_cons: self.config.nest.DEFAULT_NEST_SYNAPTIC_WEIGHT_SCALING(weight, n_cons)

        self.nest_nodes_ids = np.unique(nest_nodes_ids)
        self.tvb_simulator = tvb_simulator
        self.tvb_weights = self.tvb_connectivity.scaled_weights(mode='region')
        self.tvb_to_nest_dt_ratio = 4
        self._update_nest_dt()
        # We assume that there at least the Raw monitor which is also used for communication to/from NEST
        # If there is only the Raw monitor, then self.monitor_period = self.tvb_dt
        self.monitor_period = tvb_simulator.monitors[-1].period

        self.population_order = 100

        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations = [{"label": "E", "model": self.default_population["model"], "params": {},
                             "scale": 1, "nodes": None}]  # None means "all"

        self.default_populations_connection["delay"] = self.nest_dt
        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations_connections = \
            [{"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": 1.0,  "delay": self.default_populations_connection["delay"],
              "receptor_type": 0, "nodes": None,  # None means "all"
              }]

        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # depends on regions, set a handle to a function with
        # arguments (source_region_index=None, target_region_index=None)

        # Between NEST node delays should be at least equal to NEST time resolution
        # Therefore, zero TVB delays will become nest_dt delays in NEST
        self.default_nodes_connection["delay"] = self.default_populations_connection["delay"]
        self.nodes_connections = \
            [{"source": "E", "target": "E",
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": 1.0,  # weight scaling the TVB connectivity weight
              "delay": self.default_nodes_connection["delay"],  # additional delay to the one of TVB connectivity
              "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
            ]

        # Use these to observe NEST network behavior
        # Labels have to be different
        self.output_devices = [{"model": "spike_detector",
                                "params": self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
                                #           label <- target population
                                "connections": {"E": "E"}, "nodes": None},  # None means "all"
                               {"model": "multimeter",
                                "params": self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"],
                                #                     label <- target population
                                "connections": {"Excitatory": "E"}, "nodes": None},  # None means "all"
                               ]
        self.output_devices[1]["params"]["interval"] = self.monitor_period
        self.stimulation_devices = []  # use these for possible external stimulation devices

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
    def number_of_nest_nodes(self):
        return np.maximum(len(self.nest_nodes_ids), 1)

    @property
    def nest_nodes_labels(self):
        if len(self._nest_nodes_labels) == self.number_of_nest_nodes:
            return self._nest_nodes_labels
        else:
            return self.tvb_connectivity.region_labels[self.nest_nodes_ids.tolist()]

    def _population_property_per_node(self, property):
        output = OrderedDict()
        nest_nodes_labels = self.nest_nodes_labels
        for population in self.populations:
            output[population["label"]] = property_per_node(population[property],
                                                            population.get("nodes", self.nest_nodes_ids),
                                                            self.nest_nodes_ids, nest_nodes_labels)
        return output

    @property
    def number_of_populations(self):
        return len(self.populations)

    @property
    def populations_labels(self):
        return [population["label"] for population in self.populations]

    @property
    def populations_models(self):
        return self._population_property_per_node("model")

    @property
    def populations_nodes(self):
        return self._population_property_per_node("nodes")

    @property
    def populations_scales(self):
        return self._population_property_per_node("scales")

    @property
    def populations_sizes(self):
        scales = self._population_property_per_node("scales")
        for pop_name, scale in scales.items():
            if isinstance(scale, dict):
                for node_key, node_scale in scale.items():
                    scales[pop_name][node_key] = int(np.round(scales[pop_name][node_key] * self.population_order))
            else:
                scales[pop_name] *= self.population_order
        return scales

    @property
    def populations_params(self):
        return self._population_property_per_node("params")

    def _connection_label(self, connection):
        return "%s<-%s" % (str(connection["source"]), str(connection["target"]))

    def _connection_property_per_node(self, property, connections):
        nest_nodes_labels = self.nest_nodes_labels
        output = OrderedDict()
        for conn in connections:
            output[self._connection_label(conn)] = self._property_per_node(conn[property],
                                                                           conn.get("nodes", self.nest_nodes_ids),
                                                                           self.nest_nodes_ids, nest_nodes_labels)
        return output

    def _population_connection_property_per_node(self, property):
        return self._connection_property_per_node(property, self.populations_connections)

    @property
    def populations_connections_labels(self):
        return [self._connection_label(conn) for conn in self.populations_connections]

    @property
    def populations_connections_models(self):
        return self._population_connection_property_per_node("model")

    @property
    def populations_connections_weights(self):
        return self._population_connection_property_per_node("weight")

    @property
    def populations_connections_delays(self):
        return self._population_connection_property_per_node("delay")

    @property
    def populations_connections_receptor_types(self):
        return self._population_connection_property_per_node("receptor_type")

    @property
    def populations_connections_conn_spec(self):
        return self._population_connection_property_per_node("conn_spec")

    @property
    def populations_connections_nodes(self):
        return self._population_connection_property_per_node("nodes")

    def _nodes_connection_property_per_node(self, property, connections):
        nest_nodes_labels = self.nest_nodes_labels
        output = OrderedDict()
        for conn in connections:
            output[self._connection_label(conn)] = \
                self._property_per_nodes_connection(conn[property],
                                                    conn.get("source_nodes", self.nest_nodes_ids),
                                                    conn.get("target_nodes", self.nest_nodes_ids),
                                                    self.nest_nodes_ids, nest_nodes_labels)
        return output

    @property
    def nodes_connections_labels(self):
        return [self._connection_label(conn) for conn in self.nodes_connections]

    @property
    def nodes_connections_models(self):
        return self._nodes_connection_property_per_node("model")

    @property
    def nodes_connections_weights(self):
        return self._nodes_connection_property_per_node("weight")

    @property
    def nodes_connections_delays(self):
        return self._nodes_connection_property_per_node("delay")

    @property
    def nodes_connections_receptor_types(self):
        return self._nodes_connection_property_per_node("receptor_type")

    @property
    def nodes_connections_receptor_conn_spec(self):
        return self._nodes_connection_property_per_node("conn_spec")

    @property
    def nodes_connections_source_nodes(self):
        return self._population_connection_property_per_node("source_nodes")

    @property
    def nodes_connections_target_nodes(self):
        return self._population_connection_property_per_node("target_nodes")

    def _assert_synapse_model(self, synapse_model, delay):
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %f != 0.0 is not possible!" % delay)
            elif delay == 0.0 and synapse_model == "rate_connection_delayed":
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %f <= 0.0 is not possible!" % delay)
            elif delay == 0.0:
                return "rate_connection_instantaneous"
            else:
                return "rate_connection_delayed"
        else:
            return synapse_model

    def _assert_delay(self, delay, synapse_model="static_synapse"):
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %f != 0.0 is not possible!" % delay)
            elif synapse_model == "rate_connection_delayed" and delay <= 0.0:
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %f <= 0.0 is not possible!" % delay)
            elif delay < 0.0:
                raise_value_error("Coupling rate neurons with negative delay = %f < 0.0 is not possible!" % delay)
        elif delay <= self.nest_dt:
            LOG.warning("Coupling spiking neurons with delay = %f <= NEST integration step = %s is not possible!\n"
                        "Setting delay equal to NEST integration step!" % (delay, self.nest_dt))
            return self.nest_dt
        else:
            return delay

    def _assert_within_node_delay(self, delay):
        if delay > self.tvb_dt / 2:
            if delay > self.tvb_dt:
                raise ValueError("Within NEST node delay %f is not smaller "
                                 "than the TVB integration time step %f!"
                                 % (delay, self.tvb_dt))
            else:
                LOG.warning("Within NEST node delay %f is not smaller "
                            "than half the TVB integration time step %f!"
                            % (delay, self.tvb_dt))
        return self._assert_delay(delay)

    def _update_nest_dt(self):
        self.nest_dt = \
            float(int(np.round(self.tvb_dt / self.tvb_to_nest_dt_ratio / self.config.nest.NEST_MIN_DT))) \
            * self.config.nest.NEST_MIN_DT

    def _configure_nest_kernel(self):
        self.nest_instance.ResetKernel()  # This will restart NEST!
        self._update_nest_dt()
        self.nest_instance.set_verbosity(100)  # don't print all messages from Nest
        self.nest_instance.SetKernelStatus({"resolution": self.nest_dt, "print_time": True})

    def _confirm_compile_install_nest_models(self, models, modules=[]):
        nest_models = self.nest_instance.Models()
        models = ensure_list(models)
        modules = ensure_list(modules)
        if len(modules) == 0:
            for model in models:
                modules.append("%smodule" % model)  # Assuming default naming for modules as modelmodule
        for model, module in zip(models, cycle(modules)):
            if model not in nest_models:
                try:
                    # Try to install it...
                    self.logger.info("Trying to install module %s..." % module)
                    self.nest_instance.Install(module)
                except:
                    self.logger.info("FAILED! We need to first compile it!")
                    # ...unless we need to first compile it:
                    compile_modules(model, recompile=False, config=self.config.nest)
                    # and now install it...
                    self.logger.info("Installing now module %s..." % module)
                    self.nest_instance.Install(module)
                    self.logger.info("DONE installing module %s!" % module)
                nest_models = self.nest_instance.Models()

    def _configure_populations(self):
        # Every population must have his own model model,
        # scale of spiking neurons' number, and model specific parameters
        models = []
        for i_pop, population in enumerate(self.populations):
            temp_population = dict(self.default_population)
            temp_population.update(population)
            self.populations[i_pop] = temp_population
            if len(self.populations[i_pop].get("label", "")) == 0:
                self.populations[i_pop]["label"] = "Pop%d" % i_pop
            if self.populations[i_pop]["nodes"] is None:
                self.populations[i_pop]["nodes"] = self.nest_nodes_ids
            if hasattr(self.populations[i_pop]["model"], "__call__"):
                for node_id in self.populations[i_pop]["nodes"]:
                    models.append(self.populations[i_pop]["model"](node_id))
            else:
                models.append(self.populations[i_pop]["model"])
                self.populations[i_pop]["model"] = property_to_fun(self.populations[i_pop]["model"])
            self.populations[i_pop]["scale"] = property_to_fun(self.populations[i_pop]["scale"])
            self.populations[i_pop]["params"] = property_to_fun(self.populations[i_pop]["params"])
        models = np.unique(models)
        self._confirm_compile_install_nest_models(models)

    def _configure_connections(self, connections, default_connection):
        for i_con, connection in enumerate(connections):
            for pop in ["source", "target"]:
                if len(connection.get(pop, "")) == 0:
                    raise_value_error("No %s population in connection!:\n%s" % (pop, str(connection)))
            temp_conn = dict(default_connection)
            temp_conn.update(connection)
            connections[i_con] = temp_conn
        return connections

    def _configure_populations_connections(self):
        self.populations_connections = self._configure_connections(self.populations_connections,
                                                                   self.default_populations_connection)
        for i_conn, connections in enumerate(self.populations_connections):
            if connections["nodes"] is None:
                self.populations_connections[i_conn]["nodes"] = self.nest_nodes_ids

    def _configure_nodes_connections(self):
        self.nodes_connections = self._configure_connections(self.nodes_connections,
                                                             self.default_nodes_connection)
        for i_conn, connections in enumerate(self.nodes_connections):
            for pop in ["source", "target"]:
                this_pop = "%s_nodes" % pop
                if connections[this_pop] is None:
                    self.nodes_connections[i_conn][this_pop] = self.nest_nodes_ids
        self.tvb_connectivity.configure()

    def configure(self):
        self._configure_populations()
        self._configure_populations_connections()
        self._configure_nodes_connections()

    def _synaptic_weight_scaling(self, weights, number_of_connections):
        return self.default_synaptic_weight_scaling(weights, number_of_connections)

    def build_nest_populations(self, node_label, node_id=None):
        # Generate a NEST spiking network population...
        node = NESTRegionNode(self.nest_instance, node_label)
        for iP, population in enumerate(self.populations):
            if node_id in population["nodes"]:
                node[population["label"]] = create_population(self.nest_instance,
                                                              population["model"](node_id),
                                                              population["scale"](node_id) * self.population_order,
                                                              params=population["params"](node_id))
        return node

    def build_nest_nodes(self):
        nest_nodes_labels = self.nest_nodes_labels
        self.nodes = Series()
        for node_id, node_label in zip(self.nest_nodes_ids, nest_nodes_labels):  # For every NEST node
            # ...generate a network of spiking populations
            self.nodes[node_label] = self.build_nest_populations(node_label, node_id)

    def _connect_two_populations(self, pop_src, pop_trg, conn_spec, syn_spec):
        conn_spec, n_cons = create_connection_dict(n_src=len(pop_src), n_trg=len(pop_trg),
                                                   src_is_trg=(pop_src == pop_trg), config=self.config.nest,
                                                   **conn_spec)
        # Scale the synaptic weight with respect to the total number of connections between the two populations:
        syn_spec["weight"] = self._synaptic_weight_scaling(syn_spec["weight"], n_cons)
        syn_spec["model"] = self._assert_synapse_model(syn_spec["model"], syn_spec["delay"])
        if syn_spec["delay"] <= 0.0:
            del syn_spec["delay"]  # For instantaneous rate connections
        receptors = ensure_list(syn_spec["receptor_type"])
        for receptor in receptors:
            syn_spec["receptor_type"] = receptor
            connect_two_populations(self.nest_instance, pop_src, pop_trg, conn_spec, syn_spec)

    def _connect_two_populations_within_node(self, pop_src, pop_trg,
                                             conn_spec, syn_model, weight, delay, receptor_type):
        syn_spec = {'model': syn_model,
                    'weight': weight,
                    'delay': self._assert_within_node_delay(delay),
                    'receptor_type': receptor_type}
        self._connect_two_populations(pop_src, pop_trg, conn_spec, syn_spec)

    def connect_within_node_nest_populations(self):
        # Define a function for the exact synthesis of source and target populations
        population = lambda node, populations: \
            flatten_tuple([node[pop] for pop in ensure_list(populations)])
        # For every different type of connections between distinct NEST nodes' populations
        for i_conn, conn in enumerate(ensure_list(self.populations_connections)):
            model = property_to_fun(conn["model"])
            weight = property_to_fun(conn["weight"])
            delay = property_to_fun(conn["delay"])
            receptor_type = property_to_fun(conn["receptor_type"])
            conn_spec = property_to_fun(conn['conn_spec'])
            # ...and form the connection within each NEST node
            for node_index in conn["nodes"]:
                i_node = np.where(self.nest_nodes_ids == node_index)[0][0]
                self._connect_two_populations_within_node(population(self.nodes[i_node], conn["source"]),
                                                          population(self.nodes[i_node], conn["target"]),
                                                          conn_spec(node_index),
                                                          model(node_index),
                                                          weight(node_index),
                                                          delay(node_index),
                                                          receptor_type(node_index))

    def _connect_two_populations_between_nodes(self, pop_src, pop_trg, i_n_src, i_n_trg,
                                               conn_spec, syn_model, weight, delay, receptor_type):
        src_node_id = self.nest_nodes_ids[i_n_src]
        trg_node_id = self.nest_nodes_ids[i_n_trg]
        syn_spec = {'model': syn_model,
                    'weight': self.tvb_weights[src_node_id, trg_node_id] * weight,
                    'delay': self._assert_delay(self.tvb_delays[src_node_id, trg_node_id] + delay),
                    'receptor_type': receptor_type}
        self._connect_two_populations(pop_src, pop_trg, conn_spec, syn_spec)

    def connect_nest_nodes(self):
        # Define a function for the exact synthesis of source and target populations
        population = lambda node, populations: \
            flatten_tuple([node[pop] for pop in ensure_list(populations)])
        # For every different type of connections between distinct NEST nodes' populations
        for i_conn, conn in enumerate(ensure_list(self.nodes_connections)):
            model = property_to_fun(conn["model"])
            weight = property_to_fun(conn["weight"])
            delay = property_to_fun(conn["delay"])
            receptor_type = property_to_fun(conn["receptor_type"])
            conn_spec = property_to_fun(conn['conn_spec'])
            # ...and form the connection for every distinct pair of NEST nodes
            for source_index in conn["source_nodes"]:
                i_source_node = np.where(self.nest_nodes_ids == source_index)[0][0]
                for target_index in conn["target_nodes"]:
                    if source_index != target_index:  # TODO! Confirm that no self connections are allowed here!
                        i_target_node = np.where(self.nest_nodes_ids == target_index)[0][0]
                        if self.tvb_weights[source_index, target_index] > 0:
                            self._connect_two_populations_between_nodes(population(self.nodes[i_source_node],
                                                                                   conn["source"]),
                                                                        population(self.nodes[i_target_node],
                                                                                   conn["target"]),
                                                                        i_source_node, i_target_node,
                                                                        conn_spec(source_index, target_index),
                                                                        model(source_index, target_index),
                                                                        weight(source_index, target_index),
                                                                        delay(source_index, target_index),
                                                                        receptor_type(source_index, target_index))

    def build_and_connect_nest_stimulation_devices(self):
        # Build devices by the variable model they stimulate (Series),
        # population (Series),
        # and target node (Series) for faster reading
        stimulation_devices = Series()
        for devices in self.stimulation_devices:
            target_nest_nodes = devices.pop("nodes", self.nest_nodes_ids)
            n_target_nest_nodes = len(target_nest_nodes)
            if target_nest_nodes is None:
                target_nest_nodes = self.nest_nodes_ids
            param = property_to_fun(devices.pop("params", {}))
            weight = property_to_fun(devices.pop("weights", 1.0))
            delay = property_to_fun(devices.pop("delays", 0.0))
            receptor_type = property_to_fun(devices.pop("receptor_types", 0))
            receptor_types = np.zeros(n_target_nest_nodes, ).astype("i")
            params = np.array([{}] * n_target_nest_nodes)
            weights = 1.0 + receptor_types
            delays = self.nest_instance.GetKernelStatus("resolution") + receptor_types
            target_nest_nodes_ids = [np.where(self.nest_nodes_ids == trg_node)[0][0] for trg_node in target_nest_nodes]
            devices["nodes"] = target_nest_nodes_ids
            for trg_node, i_trg in zip(target_nest_nodes, target_nest_nodes_ids):
                params[i_trg] = param(trg_node)
                weights[i_trg] = weight(trg_node)
                delays[i_trg] = delay(trg_node)
                receptor_types[i_trg] = receptor_type(trg_node)
            devices["weights"] = weights
            devices["delays"] = delays
            devices["receptor_types"] = receptor_types
            devices["params"] = params
            stimulation_devices = \
                stimulation_devices.append(
                    build_and_connect_input_devices(self.nest_instance, devices, self.nodes))
        return stimulation_devices

    def build_and_connect_nest_output_devices(self):
        # Build devices by the variable model they measure (Series),
        # population (Series),
        # and target node (Series) for faster reading
        return build_and_connect_output_devices(self.nest_instance, self.output_devices, self.nodes)

    def build_nest_network(self):
        self._configure_nest_kernel()
        # Configure all inputs to set them to the correct formats and sizes
        self.configure()
        # Build and connect internally all NEST nodes
        self.build_nest_nodes()
        self.connect_within_node_nest_populations()
        # Connect NEST nodes among each other
        self.connect_nest_nodes()
        # Build and connect possible NEST output devices
        # !!Use it only for extra NEST quantities
        # that do not correspond to TVB state variables or parameters
        # you wish to transmit from NEST to TVB!!
        output_devices = self.build_and_connect_nest_output_devices()
        # Build and connect possible NEST input devices
        # !!Use it only for stimuli, if any!!
        stimulation_devices = self.build_and_connect_nest_stimulation_devices()
        return NESTNetwork(self.nest_instance, self.nodes, output_devices, stimulation_devices,
                           self.default_nodes_connection["delay"], self.config)


def node_key_index_and_label(node, labels):
    if isinstance(node, string_types):
        try:
            i_node = labels.index(node)
            label = node
            node_key = "%d-%s" % (node, i_node)
        except:
            raise_value_error("Node %s is not a region node modeled in NEST!" % node)
    else:
        try:
            label = labels.index(node)
            i_node = node
            node_key = "%d-%s" % (label, node)
        except:
            raise_value_error("Node %d is not a region node modeled in NEST!" % node)
    return node_key, i_node, label


def property_per_node(property, nodes, nest_nodes_labels):
    if hasattr(property, "__call__"):
        property_per_node = OrderedDict()
        for node in nodes:
            node_key, node_index = node_key_index_and_label(node, nest_nodes_labels)[:2]
            property_per_node[node_key] = property(node_index)
        return property_per_node
    else:
        return property


def property_per_nodes_connection(property, source_nodes, target_nodes, nest_nodes_ids, nest_nodes_labels):
    if hasattr(property, "__call__"):
        if source_nodes is None:
            source_nodes = nest_nodes_ids
        else:
            source_nodes = np.unique(source_nodes)
        if target_nodes is None:
            target_nodes = nest_nodes_ids
        else:
            target_nodes = np.unique(target_nodes)
        property_per_nodes_connection = OrderedDict()
        for source_node in source_nodes:
            source_index, source_label = node_key_index_and_label(source_node, nest_nodes_labels)[1:]
            for target_node in target_nodes:
                target_index, target_label = node_key_index_and_label(target_node, nest_nodes_labels)[1:]
            node_connection_label = "%d.%s<-%d.%s" % (source_index, source_label, target_index, target_label)
            property_per_nodes_connection[node_connection_label] = property(source_index, target_index)
        return property_per_nodes_connection
    else:
        return property

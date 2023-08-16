# -*- coding: utf-8 -*-

import os
from abc import ABCMeta, abstractmethod
from six import string_types
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from pandas import concat

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import CONFIGURED, initialize_logger
from tvb_multiscale.core.utils.file_utils import load_pickled_dict
from tvb_multiscale.core.utils.data_structures_utils import property_to_fun, safe_dict_copy
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import \
    serialize_tvb_cosimulator
from tvb_multiscale.core.spiking_models.brain import SpikingBrain
from tvb_multiscale.core.spiking_models.devices import DeviceSets


LOG = initialize_logger(__name__, config=CONFIGURED)


class SpikingNetworkBuilder(object):
    __metaclass__ = ABCMeta

    """This is the base class of a SpikingNetworkBuilder, 
       which builds a SpikingNetwork from user configuration inputs.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
       The builder is half way opionionated.
    """

    spiking_simulator = None
    _spiking_simulator_name = ""

    # Default configurations modifiable by the user:
    config = CONFIGURED

    default_tvb_dt = 0.1
    tvb_to_spiking_dt_ratio = config.TVB_TO_SPIKING_DT_RATIO
    default_min_spiking_dt = config.MIN_SPIKING_DT
    default_min_delay_ratio = config.MIN_DELAY_RATIO
    default_min_delay = config.DEF_SPIKING_MIN_DELAY
    default_population = {}
    default_populations_connection = {}
    default_nodes_connection = {}
    default_devices_connection = {}

    population_order = 100

    # User inputs:
    model = None
    tvb_serial_sim = None
    spiking_nodes_inds = []
    populations = []
    populations_connections = []
    nodes_connections = []
    output_devices = []  # Use these to observe Spiking Simulator behavior
    input_devices = []   # use these for possible external stimulation devices

    # Internal configurations and outputs:
    monitor_period = 1.0
    spiking_dt = default_tvb_dt / tvb_to_spiking_dt_ratio
    _spiking_nodes_labels = []
    _populations = []
    _populations_connections = []
    _nodes_connections = []
    _output_devices = []
    _input_devices = []
    _spiking_brain = SpikingBrain(name="brain regions")
    _models = []

    def __init__(self, tvb_serial_sim={}, spiking_nodes_inds=[], spiking_simulator=None, config=None, logger=None):
        self.logger = logger
        self.config = config
        self.tvb_serial_sim = tvb_serial_sim
        self.spiking_nodes_inds = spiking_nodes_inds
        self.spiking_simulator = spiking_simulator

    def __setattr__(self, attr, val):
        if attr == self._spiking_simulator_name:
            self.spiking_simulator = val
        else:
            super(SpikingNetworkBuilder, self).__setattr__(attr, val)

    def _assert_tvb_cosimulator(self):
        if isinstance(self.tvb_serial_sim, os.PathLike):
            self.tvb_serial_sim = load_pickled_dict(self.tvb_serial_sim)
        elif not isinstance(self.tvb_serial_sim, dict):
            self.tvb_serial_sim = serialize_tvb_cosimulator(self.tvb_serial_sim)

    def configure(self):
        if self.config is None:
            self.config = CONFIGURED
        if self.logger is None:
            self.logger = initialize_logger(__name__, config=self.config)
        self.tvb_to_spiking_dt_ratio = self.config.TVB_TO_SPIKING_DT_RATIO
        self.default_min_spiking_dt = self.config.MIN_SPIKING_DT
        self.default_min_delay_ratio = self.config.MIN_DELAY_RATIO
        self.default_min_delay = self.config.DEF_SPIKING_MIN_DELAY

        # Setting SpikingNetwork defaults from config
        # to be further specified in each Spiking simulator's specific builder class.
        if self.model is None:
            self.model = self.config.DEFAULT_SPIKING_MODEL
        self.default_population = {"model": self.model, "scale": 1, "params": {}, "nodes": None}

        self.default_populations_connection = safe_dict_copy(self.config.DEFAULT_CONNECTION)
        self.default_populations_connection["nodes"] = None

        self.default_nodes_connection = safe_dict_copy(self.config.DEFAULT_CONNECTION)
        self.default_nodes_connection.update({"source_nodes": None, "target_nodes": None})

        self.default_devices_connection = safe_dict_copy(self.config.DEFAULT_CONNECTION)
        self.default_devices_connection["delay"] = self.default_min_delay
        self.default_devices_connection["nodes"] = None

        self._assert_tvb_cosimulator()
        self.update_spiking_dt()
        self.update_default_min_delay()

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_serial_sim.get("coupling.a", np.array([1.0/256]))[0].item()

        # We assume that there at least the Raw monitor which is also used for communication to/from Spiking Simulator
        # If there is only the Raw monitor, then self.monitor_period = self.tvb_dt
        self.monitor_period = self.tvb_serial_sim["monitor.period"]

        self.spiking_nodes_inds = np.unique(self.spiking_nodes_inds)

    @abstractmethod
    def build_spiking_population(self, label, model, brain_region, size, params):
        """This methods builds a SpikingPopulation instance,
           which represents a population of spiking neurons of the same neural model,
           and residing at a particular brain region node.
           Arguments:
            label: name (string) of the population
            model: name (string) of the neural model
            brain_region: name (string) of the brain reegion the population will reside
            size: number (integer) of the neurons of this population
            params: dictionary of parameters of the neural model to be set upon creation
           Returns:
            a SpikingPopulation class instance
        """
        pass

    @abstractmethod
    def build_spiking_region_node(self, label="", input_node=None, *args, **kwargs):
        """This methods builds a SpikingRegionNode instance,
           which consists of all SpikingPopulation instances,
           residing at a particular brain region node.
           Arguments:
            label: name (string) of the region node. Default = ""
            input_node: an already created SpikingRegionNode() class. Default = None.
            *args, **kwargs: other optional positional or keyword arguments
           Returns:
            a SpikingRegionNode class instance
        """
        pass

    @property
    def min_delay(self):
        return self.default_min_delay

    @abstractmethod
    def set_synapse(self, syn_model, weight, delay, receptor_type, params=dict()):
        """Method to set the synaptic model, the weight, the delay,
           the synaptic receptor type, and other possible synapse parameters
           to a synapse_params dictionary.
           Arguments:
            - syn_model: the name (string) of the synapse model
            - weight: the weight of the synapse
            - delay: the delay of the connection,
            - receptor_type: the receptor type
            - params: a dict of possible synapse parameters
           Returns:
            a dictionary of the whole synapse configuration

        """
        pass

    @abstractmethod
    def connect_two_populations(self, source, src_inds_fun, target, trg_inds_fun, conn_params, synapse_params):
        """Method to connect two SpikingPopulation instances in the SpikingNetwork.
           Arguments:
            source: the source SpikingPopulation of the connection
            src_inds_fun: a function that selects a subset of the source population neurons
            target: the target SpikingPopulation of the connection
            trg_inds_fun: a function that selects a subset of the target population neurons
            conn_params: a dict of parameters of the connectivity pattern among the neurons of the two populations,
                         excluding weight and delay ones
            synapse_params: a dict of parameters of the synapses among the neurons of the two populations,
                            including weight, delay and synaptic receptor type ones
        """
        pass

    @abstractmethod
    def build_and_connect_devices(self, devices):
        """A method to build and connect to the network all devices in the input configuration dict."""
        pass

    @abstractmethod
    def build_spiking_network(self):
        """A method to build the final SpikingNetwork class based on the already created constituents."""
        pass

    @property
    def tvb_dt(self):
        if self.tvb_serial_sim is None:
            return self.default_tvb_dt
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("integrator.dt", self.default_tvb_dt)

    @property
    def tvb_model(self):
        if self.tvb_serial_sim is None:
            return ""
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("model", "")

    @property
    def number_of_regions(self):
        if self.tvb_serial_sim is None:
            return 1
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("connectivity.number_of_regions", 1)

    @property
    def region_labels(self):
        if self.tvb_serial_sim is None:
            return np.array(["%d" % reg for reg in np.arange(self.number_of_regions)])
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("connectivity.region_labels",
                                       np.array(["%d" % reg for reg in np.arange(self.number_of_regions)]))

    @property
    def tvb_weights(self):
        if self.tvb_serial_sim is None:
            return np.ones((self.number_of_regions, self.number_of_regions))
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("connectivity.weights",
                                       np.ones((self.number_of_regions, self.number_of_regions)))

    @property
    def tvb_delays(self):
        if self.tvb_serial_sim is None:
            return self.tvb_dt * np.ones((self.number_of_regions, self.number_of_regions))
        self._assert_tvb_cosimulator()
        return self.tvb_serial_sim.get("connectivity.delays",
                                       self.tvb_dt * np.ones((self.number_of_regions, self.number_of_regions)))

    @property
    def number_of_spiking_nodes(self):
        return len(self.spiking_nodes_inds)

    # The methods below are used in order to return the builder's properties
    # per spiking node or spiking nodes' connection

    @property
    def spiking_nodes_labels(self):
        if len(self._spiking_nodes_labels) == self.number_of_spiking_nodes:
            return self._spiking_nodes_labels
        else:
            return self.region_labels[self.spiking_nodes_inds]

    def _population_property_per_node(self, property):
        output = OrderedDict()
        for population in self.populations:
            output[population["label"]] = property_per_node(population[property],
                                                            population.get("nodes", self.spiking_nodes_inds),
                                                            self.region_labels)
        return output

    @property
    def number_of_populations(self):
        return len(self.populations)

    @property
    def populations_models(self):
        return self._population_property_per_node("model")

    @property
    def populations_nodes(self):
        return self._population_property_per_node("nodes")

    @property
    def populations_scales(self):
        return self._population_property_per_node("scale")

    @property
    def populations_sizes(self):
        """Method to return the number of neurons of each SpikingPopulation of the network."""
        sizes = self._population_property_per_node("scale")
        for pop_name, scale in sizes.items():
            if isinstance(scale, dict):
                for node_key, node_scale in scale.items():
                    sizes[pop_name][node_key] = int(np.round(sizes[pop_name][node_key] * self.population_order))
            else:
                sizes[pop_name] *= self.population_order
        return sizes

    @property
    def populations_params(self):
        return self._population_property_per_node("params")

    def _connection_label(self, connection):
        return "%s->%s" % (str(connection["source"]), str(connection["target"]))

    def _connection_property_per_node(self, property, connections):
        output = OrderedDict()
        for conn in connections:
            output[self._connection_label(conn)] = \
                property_per_node(conn[property], conn.get("nodes", self.spiking_nodes_inds),
                                  self.region_labels)
        return output

    def _population_connection_property_per_node(self, property):
        return self._connection_property_per_node(property, self.populations_connections)

    @property
    def populations_connections_labels(self):
        return [self._connection_label(conn) for conn in self.populations_connections]

    @property
    def populations_connections_models(self):
        return self._population_connection_property_per_node("synapse_model")

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

    def _nodes_connection_property_per_node(self, property):
        output = OrderedDict()
        for conn in self.nodes_connections:
            output[self._connection_label(conn)] = \
                property_per_nodes_connection(conn[property],
                                              conn.get("source_nodes", self.spiking_nodes_inds),
                                              conn.get("target_nodes", self.spiking_nodes_inds),
                                              self.spiking_nodes_inds, self.region_labels)
        return output

    @property
    def nodes_connections_labels(self):
        return [self._connection_label(conn) for conn in self.nodes_connections]

    @property
    def nodes_connections_models(self):
        return self._nodes_connection_property_per_node("synapse_model")

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
    def nodes_connections_conn_spec(self):
        return self._nodes_connection_property_per_node("conn_spec")

    @property
    def nodes_connections_source_nodes(self):
        return self._nodes_connection_property_per_node("source_nodes")

    @property
    def nodes_connections_target_nodes(self):
        return self._nodes_connection_property_per_node("target_nodes")

    def _info_properties(self):
        output = ""
        for prop in ["min_delay", "tvb_dt", "monitor_period", "tvb_model",
                     "number_of_regions", "tvb_weights", "tvb_delays", "region_labels",
                     "number_of_populations", "populations_models", "populations_nodes",
                     "populations_scales", "populations_sizes", "populations_params",
                     "populations_connections_labels", "populations_connections_models",
                     "populations_connections_nodes",
                     "populations_connections_weights", "populations_connections_delays",
                     "populations_connections_receptor_types", "populations_connections_conn_spec",
                     "nodes_connections_labels", "nodes_connections_models",
                     "nodes_connections_source_nodes", "nodes_connections_target_nodes",
                     "nodes_connections_weights", "nodes_connections_delays", "nodes_connections_receptor_types",
                     "nodes_connections_conn_spec"]:
            output += "\n%s:\n%s\n" % (prop, str(getattr(self, prop, None)))
        return output

    def info(self):
        return self.__str__() + self._info_properties()

    def _assert_delay(self, delay):
        assert delay >= 0.0
        return delay

    def _assert_within_node_delay(self, delay):
        # TODO: decide about default constraints to minimum delays of the network!
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

    def update_spiking_dt(self):
        # The TVB dt should be an integer multiple of the spiking simulator dt:
        self.spiking_dt = int(np.round(self.tvb_dt / self.tvb_to_spiking_dt_ratio / self.default_min_spiking_dt)) \
                          * self.default_min_spiking_dt

    def update_default_min_delay(self):
        # The Spiking Network min delay should be smaller than half the TVB dt,
        # and an integer multiple of the spiking simulator dt
        self.default_min_delay = np.minimum(
            np.maximum(self.default_min_delay_ratio * self.spiking_dt, self.min_delay),
            self.tvb_dt / self.tvb_to_spiking_dt_ratio)

    def _configure_populations(self):
        # Every population must have its own model model, label.
        # scale of spiking neurons' number, and model specific parameters,
        # and a list of spiking region nodes where it is going to be placed
        # "scale" and "parameters" can be given as functions.
        # This configuration will confirm user inputs
        # and set the two properties above as functions of node index.
        self.populations_labels = list()
        _populations = list()
        for i_pop, population in enumerate(self.populations):
            _populations.append(deepcopy(self.default_population))
            _populations[-1].update(population)
            if len(_populations[-1].get("label", "")) == 0:
                _populations[-1]["label"] = "Pop%d" % i_pop
            self.populations_labels.append(_populations[-1]["label"])
            if _populations[-1]["nodes"] is None:
                _populations[-1]["nodes"] = self.spiking_nodes_inds
            else:
                _populations[-1]["nodes"] = np.array(ensure_list(_populations[-1]["nodes"]))
            _model = _populations[-1]["model"]
            if _model not in self._models:
                self._models.append(_model)
            _populations[-1]["scale"] = property_to_fun(_populations[-1]["scale"])
            _populations[-1]["params"] = property_to_fun(_populations[-1]["params"])
        self.populations_labels = np.unique(self.populations_labels).tolist()
        self._populations = _populations
        return self._populations

    def _assert_connection_populations(self, connection):
        # This method will make sure that there are source and target user inputs for every population connection
        # and that every source/target population is already among the populations to be generated.
        for pop in ["source", "target"]:
            pops_labels = connection.get(pop, None)
            if pops_labels is None:
                raise_value_error("No %s population in connection!:\n%s" % (pop, str(connection)))
            for pop_lbl in ensure_list(pops_labels):
                assert pop_lbl in self.populations_labels

    def _configure_connections(self, connections, default_connection):
        # This method sets "weight", "delay" and "receptor_type" synapse properties,
        # as well as a possible synaptic "params" dictionary,
        # as functions of the node where the populations are placed
        _connections = []
        for i_con, connection in enumerate(connections):
            self._assert_connection_populations(connection)
            temp_conn = safe_dict_copy(default_connection)
            temp_conn.update(connection)
            _connections.append(temp_conn)
            for prop in ["weight", "delay", "receptor_type", "params"]:
                _connections[i_con][prop] = property_to_fun(_connections[i_con][prop])
            for prop in ["source_neurons", "target_neurons"]:
                inds_fun = _connections[i_con].get(prop, None)
                if inds_fun is not None:
                    _connections[i_con][prop] = property_to_fun(inds_fun)
                else:
                    _connections[i_con][prop] = None
            _model = _connections[i_con].get("synapse_model", _connections[i_con].get("model", None))
            if _model is not None and _model not in self._models:
                self._models.append(_model)
        return _connections

    def _configure_populations_connections(self):
        # This method configures the within node populations' connections,
        # assuming their creation at all spiking region nodes, unless a different input configuration
        # is provided by the user.
        _populations_connections = self._configure_connections(self.populations_connections,
                                                               self.default_populations_connection)
        for i_conn, connections in enumerate(self.populations_connections):
            if connections.get("nodes", None) is None:
                _populations_connections[i_conn]["nodes"] = self.spiking_nodes_inds
            else:
                _populations_connections[i_conn]["nodes"] = \
                    np.array(ensure_list(_populations_connections[i_conn]["nodes"]))
        self._populations_connections = _populations_connections
        return self._populations_connections

    def _configure_nodes_connections(self):
        # This method configures the connections among region nodes,
        # assuming their creation among all pairs of spiking region nodes,
        # unless a different input configuration is provided by the user,
        # for the source and/or target region nodes.
        _nodes_connections = self._configure_connections(self.nodes_connections,
                                                         self.default_nodes_connection)
        for i_conn, connections in enumerate(self.nodes_connections):
            for pop in ["source", "target"]:
                this_pop = "%s_nodes" % pop
                if connections.get(this_pop, None) is None:
                    _nodes_connections[i_conn][this_pop] = self.spiking_nodes_inds
                else:
                    _nodes_connections[i_conn][this_pop] = np.array(ensure_list(_nodes_connections[i_conn][this_pop]))
        self._nodes_connections = _nodes_connections
        return self._nodes_connections

    def _configure_devices(self, devices):
        # Configure devices by
        # the variable model they measure or stimulate, and population(s) they target
        # and brain region node where they refer to.
        # "weight", "delay" and "receptor_type" are set as functions, following user input
        _devices = list()
        for device in devices:
            _devices.append(safe_dict_copy(device))
            spiking_nodes = device.get("nodes", None)
            if spiking_nodes is None:
                spiking_nodes = self.spiking_nodes_inds
            else:
                spiking_nodes = np.array(ensure_list(spiking_nodes))
                assert np.all([trg_node in self.spiking_nodes_inds for trg_node in spiking_nodes])
            # User inputs
            # ..set/converted to functions
            weights_fun = property_to_fun(device.get("weights",
                                                     self.default_devices_connection["weight"]))
            delays_fun = property_to_fun(device.get("delays",
                                                    self.default_devices_connection["delay"]))
            receptor_type_fun = property_to_fun(device.get("receptor_type",
                                                           self.default_devices_connection["receptor_type"]))
            syn_spec_fun = property_to_fun(device.get("syn_spec", None))
            conn_spec_fun = property_to_fun(device.get("conn_spec", None))
            # Default behavior for any region nodes is to target all of the populations' neurons:
            neurons_fun = device.get("neurons_fun", None)
            if neurons_fun is not None:
                neurons_fun = property_to_fun(neurons_fun)
            # Defaults in arrays:
            shape = (len(spiking_nodes),)
            receptor_type = np.tile(self.default_devices_connection["receptor_type"], shape).astype("O")
            # weights and delays might be dictionaries for distributions:
            weights = np.tile(self.default_devices_connection["weight"], shape).astype("O")
            delays = np.tile(self.default_devices_connection["delay"], shape).astype("O")
            neurons = np.tile([None], shape).astype("O")
            syn_spec = np.tile([None], shape).astype("O")
            conn_spec = np.tile([None], shape).astype("O")
            # Set now the properties using the above defined functions:
            for i_trg, trg_node in enumerate(spiking_nodes):
                weights[i_trg] = weights_fun(trg_node)  # a function also of self.tvb_weights
                delays[i_trg] = delays_fun(trg_node)    # a function also of self.tvb_delays
                receptor_type[i_trg] = receptor_type_fun(trg_node)
                syn_spec[i_trg] = syn_spec_fun(trg_node)
                conn_spec[i_trg] = conn_spec_fun(trg_node)
                if neurons_fun is not None:
                    neurons[i_trg] = lambda neurons: neurons_fun(trg_node, neurons)
            _devices[-1]["params"] = device.get("params", {})
            _devices[-1]["weights"] = weights
            _devices[-1]["delays"] = delays
            _devices[-1]["receptor_type"] = receptor_type
            _devices[-1]["syn_spec"] = syn_spec
            _devices[-1]["conn_spec"] = conn_spec
            _devices[-1]["neurons_fun"] = neurons
            _devices[-1]["nodes"] = [np.where(self.spiking_nodes_inds == trg_node)[0][0] for trg_node in spiking_nodes]
        return _devices

    def _configure_output_devices(self):
        self._output_devices = self._configure_devices(self.output_devices)
        return self._output_devices

    def _configure_input_devices(self):
        self._input_devices = self._configure_devices(self.input_devices)
        return self._input_devices

    def _configure(self):
        """Method to configure the builder taking into consideration the input configurations by the user."""
        LOG.info("Configuring populations...")
        self._configure_populations()
        LOG.info("Configuring populations' connections within spiking region nodes...")
        self._configure_populations_connections()
        LOG.info("Configuring populations' connections among spiking region nodes...")
        self._configure_nodes_connections()
        LOG.info("Configuring output devices, if any...")
        self._configure_output_devices()
        LOG.info("Configuring input devices, if any...")
        self._configure_input_devices()

    def build_spiking_region_nodes(self, *args, **kwargs):
        """Method to build all spiking populations with each brain region node."""
        # For every Spiking node
        for node_id, node_label in zip(self.spiking_nodes_inds, self.spiking_nodes_labels):
            LOG.info("Generating spiking region node: %s..." % node_label)
            self._spiking_brain[node_label] = self.build_spiking_region_node(node_label)
            # ...and every population in it...
            for iP, population in enumerate(self._populations):
                # ...if this population exists in this node...
                if node_id in population["nodes"]:
                    LOG.info("Generating population: %s..." % population["label"])
                    # ...generate this population in this node...
                    size = int(np.round(population["scale"](node_id) * self.population_order))
                    self._spiking_brain[node_label][population["label"]] = \
                        self.build_spiking_population(population["label"], population["model"], node_label, size,
                                                      params=population["params"](node_id),
                                                      *args, **kwargs)

    def connect_within_node_spiking_populations(self):
        """Method to connect all populations withing each Spiking brain region node."""
        # For every different type of connections between distinct Spiking nodes' populations
        for i_conn, conn in enumerate(ensure_list(self._populations_connections)):
            LOG.info("Connecting %s -> %s populations \nfor spiking region nodes\n%s..." %
                             (str(conn["source"]), str(conn["target"]), str(conn["nodes"])))
            # ...and for every brain region node where this connection will be created:
            for node_index in conn["nodes"]:
                i_node = np.where(self.spiking_nodes_inds == node_index)[0][0]
                # ...create a synapse parameters dictionary, from the configured inputs:
                syn_spec = self.set_synapse(conn["synapse_model"],
                                            conn['weight'](node_index),
                                            self._assert_delay(conn['delay'](node_index)),
                                            conn['receptor_type'](node_index),
                                            conn["params"](node_index)
                                            )
                # ...and for every combination of source...
                for pop_src in ensure_list(conn["source"]):
                    # ...and target populations of this connection...
                    for pop_trg in ensure_list(conn["target"]):
                        # ...connect the two populations:
                        self.connect_two_populations(
                            self._spiking_brain[i_node][pop_src], conn["source_neurons"],
                            self._spiking_brain[i_node][pop_trg], conn["target_neurons"],
                            conn["conn_spec"], syn_spec
                        )

    def connect_spiking_region_nodes(self):
        """Method to connect all Spiking brain region nodes among them."""
        # For every different type of connections between distinct Spiking region nodes' populations
        for i_conn, conn in enumerate(ensure_list(self._nodes_connections)):
            # ...form the connection for every distinct pair of Spiking nodes
            for source_index in conn["source_nodes"]:
                # ...get the source spiking brain region indice:
                i_source_node = np.where(self.spiking_nodes_inds == source_index)[0][0]
                for target_index in conn["target_nodes"]:
                    # ...get the target spiking brain region indice:
                    i_target_node = np.where(self.spiking_nodes_inds == target_index)[0][0]
                    # ...create a synapse parameters dictionary, from the configured inputs:
                    syn_spec = self.set_synapse(conn["synapse_model"],
                                                conn["weight"](source_index, target_index),
                                                conn["delay"](source_index, target_index),
                                                conn["receptor_type"](source_index, target_index)
                                                )
                    if source_index != target_index:
                        LOG.info("Connecting for %s -> %s spiking region nodes ..." %
                                         (str(source_index), str(target_index)))
                        # ...and as long as this is not a within node connection...
                        for conn_src in ensure_list(conn["source"]):
                            # ...and for every combination of source...
                            src_pop = self._spiking_brain[i_source_node][conn_src]
                            for conn_trg in ensure_list(conn["target"]):
                                # ...and target population...
                                trg_pop = self._spiking_brain[i_target_node][conn_trg]
                                LOG.info("%s -> %s populations ..." %
                                                 (src_pop.label, trg_pop.label))
                                self.connect_two_populations(src_pop, conn["source_neurons"],
                                                             trg_pop, conn["target_neurons"],
                                                             conn['conn_spec'], syn_spec)

    def build_spiking_brain(self):
        """Method to build and connect all Spiking brain region nodes,
           first withing, and then, among them.
        """
        LOG.info("Generating spiking brain regions...")
        self.build_spiking_region_nodes()
        LOG.info("Connecting populations within spiking brain regions...")
        self.connect_within_node_spiking_populations()
        # Connect Spiking nodes among each other
        LOG.info("Connecting populations among spiking brain regions...")
        self.connect_spiking_region_nodes()

    def _build_and_connect_devices(self, devices, label):
        """Method to build and connect input or output devices, organized by
           - the variable they measure or stimulate, and the
           - population(s), and
           - brain region nodes they target."""
        _devices = []
        for device in devices:
            LOG.info("Generating and connecting %s -> %s device set of model %s\n"
                     "for nodes %s..." % (str(list(device["connections"].keys())),
                                          str(list(device["connections"].values())),
                                          device["model"], str(device["nodes"])))
            _devices.append(self.build_and_connect_devices(device))
        if len(_devices):
            return DeviceSets(concat(_devices), name=label)
        else:
            return DeviceSets(name=label)

    def build_and_connect_output_devices(self):
        """Method to build and connect output devices, organized by
          - the variable they measure, and the
          - population(s), and
          - brain region nodes they target."""
        return self._build_and_connect_devices(self._output_devices, "output_devices")

    def build_and_connect_input_devices(self):
        """Method to build and connect input devices, organized by
           - the variable they stimulate, and the
           - population(s), and
           - brain region nodes they target."""
        return self._build_and_connect_devices(self._input_devices, "input_devices")

    def build(self):
        """This method will run the whole workflow of building the spiking network, which will be returned."""
        # Configure all inputs/configurations for building
        LOG.info("Configuring spiking network builder...")
        self._configure()
        # Build and connect the brain network
        LOG.info("Generating spiking brain...")
        self.build_spiking_brain()
        # Build and connect possible Spiking output devices
        # !!Use it only for extra Spiking quantities
        # that do not correspond to TVB state variables or parameters
        # you wish to transmit from the Spiking simulator to TVB!!
        LOG.info("Generating and connecting output devices, if any...")
        self._output_devices = self.build_and_connect_output_devices()
        # Build and connect possible Spiking input devices
        # !!Use it only for stimuli, if any, not for transmitting data from TVB to the Spiking simulator!!
        LOG.info("Generating and connecting input devices, if any...")
        self._input_devices = self.build_and_connect_input_devices()
        return self.build_spiking_network()


def node_key_index_and_label(node, labels):
    if isinstance(node, string_types):
        try:
            i_node = labels.index(node)
            label = node
            node_key = "%d-%s" % (i_node, node)
        except:
            raise_value_error("Node %s is not a region node modeled in Spiking Simulator!" % node)
    else:
        try:
            label = labels[node]
            i_node = node
            node_key = "%d-%s" % (node, label)
        except:
            raise_value_error("Node %d is not a region node modeled in Spiking Simulator!" % node)
    return node_key, i_node, label


# The functions below are used in order to return the builder's properties
# per spiking node or spiking nodes' connection


def property_per_node(property, nodes, nodes_labels):
    if hasattr(property, "__call__") and nodes:
        property_per_node = OrderedDict()
        for node in nodes:
            node_key, node_index = node_key_index_and_label(node, nodes_labels)[:2]
            property_per_node[node_key] = property(node_index)
        return property_per_node
    else:
        return property


def property_per_nodes_connection(property, source_nodes, target_nodes, spiking_nodes_inds, nodes_labels):
    if hasattr(property, "__call__"):
        if source_nodes is None:
            source_nodes = spiking_nodes_inds
        else:
            source_nodes = np.unique(source_nodes)
        if target_nodes is None:
            target_nodes = spiking_nodes_inds
        else:
            target_nodes = np.unique(target_nodes)
        property_per_nodes_connection = OrderedDict()
        for source_node in source_nodes:
            source_index, source_label = node_key_index_and_label(source_node, nodes_labels)[1:]
            for target_node in target_nodes:
                target_index, target_label = node_key_index_and_label(target_node, nodes_labels)[1:]
                node_connection_label = "%d.%s->%d.%s" % (source_index, source_label, target_index, target_label)
                property_per_nodes_connection[node_connection_label] = property(source_index, target_index)
        return property_per_nodes_connection
    else:
        return property

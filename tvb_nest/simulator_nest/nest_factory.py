# -*- coding: utf-8 -*-

import os
import shutil
from six import string_types
from pandas import Series
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models.devices import NESTInputDeviceDict, NESTOutputDeviceDict, NESTDeviceSet
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import flatten_tuple, ensure_list


LOG = initialize_logger(__name__)


# Helper functions with NEST

def log_path(name, logger=LOG):
    logger.info("%s: %s" % (name, os.environ.get(name, "")))


def load_spiking_simulator(logger=LOG):

    logger.info("Loading a NEST instance...")
    log_path('NEST_PYTHON_PREFIX', logger)
    log_path('PYTHONPATH', logger)
    log_path('PATH', logger)
    log_path('NEST_INSTALL_DIR', logger)

    import nest
    return nest


def compile_modules(modules, recompile=False, config=CONFIGURED, logger=LOG):
    # ...unless we need to first compile it:
    from pynestml.frontend.pynestml_frontend import install_nest
    if not os.path.exists(config.MODULES_BLDS_DIR):
        logger.info("Creating MODULES_BLDS_DIR: %s" % config.MODULES_BLDS_DIR)
        os.makedirs(config.MODULES_BLDS_DIR)
    for module in ensure_list(modules):
        logger.info("Compiling %s..." % module)
        module_bld_dir = os.path.join(config.MODULES_BLDS_DIR, module)
        logger.info("from in build directory %s..." % module_bld_dir)
        if not os.path.exists(module_bld_dir) or recompile:
            source_path = os.path.join(config.MODULES_DIR, module)
            logger.info("copying sources from %s\ninto %s..." % (source_path, module_bld_dir))
            shutil.copytree(source_path, module_bld_dir)
        logger.info("Running compilation...")
        install_nest(module_bld_dir, config.NEST_PATH)
        logger.info("DONE compiling %s!" % module)


def create_population(nest_instance, model, size, params={}):
    return nest_instance.Create(model, int(np.round(size)), params=params)


def connect_two_populations(nest_instance, pop_src, pop_trg, conn_spec={}, syn_spec={}):
    return nest_instance.Connect(pop_src, pop_trg, conn_spec, syn_spec)


def create_connection_dict(n_src=1, n_trg=1, src_is_trg=False, config=CONFIGURED, **kwargs):
    # This function returns a conn_spec dictionary
    # and the expected/accurate number of total connections
    conn_spec = dict(config.DEFAULT_CONNECTION["conn_spec"])
    P_DEF = conn_spec["p"]
    conn_spec.update(kwargs)
    rule = conn_spec["rule"]
    p = conn_spec["p"]
    N = conn_spec["N"]
    autapses = conn_spec["autapses"]
    multapses = conn_spec["multapses"]
    indegree = conn_spec["indegree"]
    outdegree = conn_spec["outdegree"]
    conn_spec = {
        'rule': rule,
        'autapses': autapses,  # self-connections flag
        'multapses': multapses  # multiple connections per neurons' pairs flag
    }
    if rule == 'one_to_one':
        # TODO: test whether there is an error
        # if Nsrc != Ntrg in this case
        # and if src_is_trg and autapses or multapses play a role
        return conn_spec, np.minimum(n_src, n_trg)
    elif rule == 'fixed_total_number':
        if N is None:
            # Assume all to all if N is not given:
            N = n_src * n_trg
            if p is not None:
                # ...prune to end up to connection probability p if p is given
                N = int(np.round(p * N))
        conn_spec['N'] = N
        return conn_spec, N
    elif rule == 'fixed_indegree':
        if indegree is None:
            # Compute indegree following connection probability p if not given
            if p is None:
                p = P_DEF
            indegree = int(np.round(p * n_src))
        conn_spec['indegree'] = indegree
        return conn_spec, indegree * n_trg
    elif rule == 'fixed_outdegree':
        if outdegree is None:
            # Compute outdegree following connection probability p if not given
            if p is None:
                p = P_DEF
            outdegree = int(np.round(p * n_trg))
        conn_spec['outdegree'] = outdegree
        return conn_spec, outdegree * n_src
    else:
        Nall = n_src * n_trg
        if src_is_trg and autapses is False:
            Nall -= n_src
        if rule == 'pairwise_bernoulli':
            if p is None:
                p = P_DEF
            conn_spec['p'] = p
            return conn_spec, int(np.round(p * Nall))
        else:  # assuming rule == 'all_to_all':
            return conn_spec, Nall


def get_neurons_from_populations(population, index=None):
    try:
        # In case this is a list, tuple or dictionary (in which case index is a key)...
        return flatten_tuple(population[index])
    except:
        # Otherwise, if index is None, return the whole tuple of neurons' indices...
        return flatten_tuple(population)


def device_to_dev_model(device):
    if device == "spike_multimeter":
        return "multimeter"
    else:
        return device


def build_output_device(nest_instance, device, config=CONFIGURED):
    from six import string_types
    if isinstance(device, string_types) or isinstance(device, dict):
        if isinstance(device, string_types):
            # If the input is only a string of the type of the device, use the default options
            dev_model = device_to_dev_model(device)
            try:
                return NESTOutputDeviceDict[device](
                    nest_instance,
                    nest_instance.Create(dev_model,
                                         params=dict(config.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(device, {}))))
            except:
                raise ValueError("Failed to set input device%s!" % str(device))
        else:
            # If there is a device dictionary...
            try:
                # ...assert the type...
                dev_name = device.get("model", None)
                dev_model = device_to_dev_model(dev_name)
                # ...assert and configure the target NEST nodes
                params = dict(config.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(dev_name, {}))
                params.update(device.get("params", {}))
                return NESTOutputDeviceDict[dev_name](
                    nest_instance,
                    nest_instance.Create(dev_model, params=params))
            except:
                raise ValueError("Failed to set output device%s!" % str(device))
    else:
        raise ValueError("Failed to set output device%s!\n "
                         "Input device has to be a device model or dict!" % str(device))


def build_input_device(nest_instance, device):
    try:
        dev_model = device["model"]
        return NESTInputDeviceDict[dev_model](
            nest_instance,
            nest_instance.Create(dev_model, params=dict(device.get("params", {}))))
    except:
        raise ValueError("Failed to set input device %s!" % str(device))


def connect_device(nest_instance, nest_device, neurons, weight=1.0, delay=0.0, receptor_type=0):
    delay = np.maximum(delay, nest_instance.GetKernelStatus("resolution"))
    if nest_device.model == "spike_detector":
        #                     source  ->  target
        nest_instance.Connect(neurons, nest_device.device,
                              syn_spec={"weight": weight, "delay": delay, "receptor_type": receptor_type})
    else:
        nest_instance.Connect(nest_device.device, neurons,
                              syn_spec={"weight": weight, "delay": delay, "receptor_type": receptor_type})
    # nest_device.update_number_of_connections()
    return nest_device


def build_and_connect_output_device(nest_instance, device, neurons, weight=1.0, delay=0.0):
    return connect_device(nest_instance, build_output_device(nest_instance, device), neurons, weight, delay)


def build_and_connect_input_device(nest_instance, device, neurons, weight=1.0, delay=0.0, receptor_type=0):
    return connect_device(nest_instance, build_input_device(nest_instance, device),
                          neurons, weight, delay, receptor_type)


def _get_connections(device, nest_nodes):
    # Determine the connections from variables to measure/stimulate to NEST node populations
    connections = device["connections"]  # either a variable model or a dict
    if isinstance(connections, string_types):
        connections = {connections: slice(None)}  # return all population types
    device_target_nodes = device.get("nodes", None)
    if device_target_nodes is None:
        device_target_nodes = nest_nodes
    else:
        device_target_nodes = nest_nodes[device_target_nodes]
    return connections, device_target_nodes


def _get_device_props_with_correct_shape(device, shape):
    dummy = np.ones(shape).astype("i")
    params = device.get("params", {})
    if isinstance(params, dict):
        params = np.tile(params, dummy.shape)
    else:
        params = np.array(params)
        if params.shape != shape:
            if params.size == 0:
                params = np.tile(params, shape)
            else:
                raise_value_error("Device parameters are neither of shape (n_devices, n_nodes) = %s"
                                  "nor of size 1:\n%s" % (str(shape), str(params)))
    weights = device.get("weights", 1.0) * dummy
    delays = device.get("delays", 0.0) * dummy
    if device["model"] in NESTInputDeviceDict.keys():
        receptor_types = device.get("receptor_types", 0) * dummy
    else:
        receptor_types = 0 * dummy
    return params, weights, delays, receptor_types


def build_and_connect_devices_one_to_one(nest_instance, device, nest_nodes):
    # This function is mostly used when a measuring (output) device targets one and only NEST node,
    # as it is the case for measuring NEST populations from specific TVB nodes
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    nest_devices = Series()
    if device["model"] in NESTInputDeviceDict.keys():
        build_device_and_connect_device = \
            lambda nest_instance, device, populations, weight=1.0, delay=0.0, receptor_type=0: \
                build_and_connect_input_device(nest_instance, device, populations, weight, delay, receptor_type)
    elif device["model"] in NESTOutputDeviceDict.keys():
        build_device_and_connect_device = \
            lambda nest_instance, device, populations, weight=1.0, delay=0.0, receptor_type=0: \
                build_and_connect_output_device(nest_instance, device, populations, weight, delay)
    else:
        raise ValueError("Device model %s is neither one of available input devices:\n%s\n"
                         "nor of output ones!:\n%s"
                         % (device["model"], NESTInputDeviceDict.keys(), NESTOutputDeviceDict.keys()))
    # Determine the connections from variables to measure/stimulate to NEST node populations
    connections, device_target_nodes = _get_connections(device, nest_nodes)
    # Determine the device's parameters and connections' properties
    params, weights, delays, receptor_types = _get_device_props_with_correct_shape(device, (len(device_target_nodes),))
    # For every NEST population variable to be stimulated or measured...
    for pop_var, populations in connections.items():
        # This set of devices will be for variable pop_var...
        nest_devices[pop_var] = NESTDeviceSet(pop_var, device["model"])
        # and for every target region node...
        for i_node, node in enumerate(device_target_nodes):
            # and for every target node and population group...
            # create a device
            nest_devices[pop_var][node.label] = \
                build_device_and_connect_device(nest_instance, device, node[populations],
                                                weights[i_node], delays[i_node], receptor_types[i_node])
    return nest_devices


def build_and_connect_devices_one_to_many(nest_instance, device, nest_nodes, names):
    # This function is mostly used when a stimulation (input) device targets more than one NEST node,
    # as it is the case for TVB state variables-per-node proxies
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    nest_devices = Series()
    if device["model"] in NESTInputDeviceDict.keys():
        build_device = lambda nest_instance, device: build_input_device(nest_instance, device)
    elif device["model"] in NESTOutputDeviceDict.keys():
        build_device = lambda nest_instance, device: build_output_device(nest_instance, device)
    else:
        raise ValueError("Device model %s is neither one of available input devices:\n%s\n"
                         "nor of output ones!:\n%s"
                         % (device["model"], NESTInputDeviceDict.keys(), NESTOutputDeviceDict.keys()))
    # Determine the connections from variables to measure/stimulate to NEST node populations
    connections, device_target_nodes = _get_connections(device, nest_nodes)
    # Determine the device's parameters and connections' properties
    params, weights, delays, receptor_types = \
        _get_device_props_with_correct_shape(device, (len(names), len(device_target_nodes)))
    # For every NEST population variable to be stimulated or measured...
    for pop_var, populations in connections.items():
        # This set of devices will be for variable pop_var...
        nest_devices[pop_var] = NESTDeviceSet(pop_var, device["model"])
        # and for every target region node...
        for i_dev, dev_name in enumerate(names):
            # and for every target node and population group...
            # create a device
            nest_devices[pop_var][dev_name] = build_device(nest_instance, device)
            for i_node, node in enumerate(device_target_nodes):
                nest_devices[pop_var][dev_name] = \
                    connect_device(nest_instance, nest_devices[pop_var][dev_name], node[populations],
                                   weights[i_dev, i_node], delays[i_dev, i_node], receptor_types[i_dev, i_node])
    return nest_devices


def build_and_connect_devices(nest_instance, devices, nest_nodes):
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    nest_devices = Series()
    for device in ensure_list(devices):
        # For every distinct quantity to be measured from NEST or stimulated towards NEST nodes...
        dev_names = device.get("names", None)
        if dev_names is None:  # If no devices' names are given...
            nest_devices = nest_devices.append(
                                build_and_connect_devices_one_to_one(nest_instance, device, nest_nodes)
                                              )
        else:
            nest_devices = nest_devices.append(
                                build_and_connect_devices_one_to_many(nest_instance, device, nest_nodes, dev_names)
                                              )
    return nest_devices


def build_and_connect_output_devices(nest_instance, devices_args, nest_nodes):
    return build_and_connect_devices(nest_instance, devices_args, nest_nodes)


def build_and_connect_input_devices(nest_instance, devices_args, nest_nodes):
    return build_and_connect_devices(nest_instance, devices_args, nest_nodes)

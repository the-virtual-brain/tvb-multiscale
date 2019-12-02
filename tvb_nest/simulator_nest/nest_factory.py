# -*- coding: utf-8 -*-

import os
import sys
import shutil
from pandas import Series
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models.devices import NESTInputDeviceDict, NESTOutputDeviceDict, NESTDeviceSet
from tvb_scripts.utils.data_structures_utils import flatten_tuple, ensure_list


# Helper functions with NEST


def load_nest(config=CONFIGURED):
    nest_path = config.nest.NEST_PATH
    os.environ['NEST_INSTALL_DIR'] = nest_path
    os.environ['PATH'] = os.path.join(nest_path, "bin") + ":" + os.environ['PATH']

    os.environ['NEST_PYTHON_PREFIX'] = config.nest.PYTHON
    sys.path.insert(0, os.environ['NEST_PYTHON_PREFIX'])

    import nest
    return nest


def compile_modules(modules, recompile=False, config=CONFIGURED):
    # ...unless we need to first compile it:
    from pynestml.frontend.pynestml_frontend import install_nest
    if not os.path.exists(config.MODULES_BLDS_DIR):
        os.makedirs(config.MODULES_BLDS_DIR)
    for module in ensure_list(modules):
        module_bld_dir = os.path.join(config.MODULES_BLDS_DIR, module)
        if not os.path.exists(module_bld_dir) or recompile:
            shutil.copytree(os.path.join(config.MODULES_DIR, module), module_bld_dir)
        install_nest(module_bld_dir, config.nest.NEST_PATH)


def create_connection_dict(n_src=1, n_trg=1, src_is_trg=False, config=CONFIGURED, **kwargs):
    # This function returns a conn_spec dictionary
    # and the expected/accurate number of total connections
    params = config.nest.DEFAULT_CONNECTION["params"]
    P_DEF = params["p"]
    params.update(kwargs)
    rule = params["rule"]
    p = params["p"]
    N = params["N"]
    autapses = params["autapses"]
    multapses = params["multapses"]
    indegree = params["indegree"]
    outdegree = params["outdegree"]
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
        conn_spec.update({'N': N})
        return conn_spec, N
    elif rule == 'fixed_indegree':
        if indegree is None:
            # Compute indegree following connection probability p if not given
            if p is None:
                p = P_DEF
            indegree = int(np.round(p * n_src))
        conn_spec.update({'indegree': indegree})
        return conn_spec, indegree * n_trg
    elif rule == 'fixed_outdegree':
        if outdegree is None:
            # Compute outdegree following connection probability p if not given
            if p is None:
                p = P_DEF
            outdegree = int(np.round(p * n_trg))
        conn_spec.update({'outdegree': outdegree})
        return conn_spec, outdegree * n_src
    else:
        Nall = n_src * n_trg
        if src_is_trg and autapses is False:
            Nall -= n_src
        if rule == 'pairwise_bernoulli':
            if p is None:
                p = P_DEF
            conn_spec.update({'p': p})
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


def build_nest_output_device(nest_instance, device, config=CONFIGURED):
    from six import string_types
    if isinstance(device, string_types) or isinstance(device, dict):
        if isinstance(device, string_types):
            # If the input is only a string of the type of the device, use the default options
            dev_model = device_to_dev_model(device)
            try:
                return NESTOutputDeviceDict[device](
                    nest_instance,
                    nest_instance.Create(dev_model,
                                         params=config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(device, {})))
            except:
                raise ValueError("Failed to set input device%s!" % str(device))
        else:
            # If there is a device dictionary...
            try:
                # ...assert the type...
                dev_name = device.get("model", None)
                dev_model = device_to_dev_model(dev_name)
                # ...assert and configure the target NEST nodes
                params = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(dev_name, {})
                params.update(device.get("params", {}))
                return NESTOutputDeviceDict[dev_name](
                    nest_instance,
                    nest_instance.Create(dev_model, params=params))
            except:
                raise ValueError("Failed to set output device%s!" % str(device))
    else:
        raise ValueError("Failed to set output device%s!\n "
                         "Input device has to be a device model or dict!" % str(device))


def build_and_connect_output_device(nest_instance, device, neurons):
    nest_out_device = build_nest_output_device(nest_instance, device)
    if nest_out_device.model == "spike_detector":
        nest_instance.Connect(neurons, nest_out_device.device)
    else:
        nest_instance.Connect(nest_out_device.device, neurons)
    nest_out_device.update_number_of_connections()
    return nest_out_device


def build_input_device(nest_instance, device):
    try:
        dev_model = device["model"]
        return NESTInputDeviceDict[dev_model](
            nest_instance,
            nest_instance.Create(dev_model, params=device.get("params", {})))
    except:
        raise ValueError("Failed to set input device %s!" % str(device))


def build_and_connect_input_device(nest_instance, device, neurons):
    nest_in_device = build_input_device(nest_instance, device)
    min_delay = nest_instance.GetKernelStatus("min_delay")
    try:
        nest_instance.Connect(nest_in_device.device, neurons,
                              syn_spec={"weight": device.get("weight", 1.0),
                                        "delay": np.maximum(min_delay, device.get("delay", 0.0))})
        nest_in_device.update_number_of_connections()
    except:
        raise ValueError("Failed to connect input device %s to populations %s!"
                         % (str(nest_in_device), str(neurons)))
    return nest_in_device


def build_and_connect_devices(nest_instance, devices, nest_nodes):
    from six import string_types
    # Build devices by their model (IndexedOrderedDict),
    # target nodes (IndexedOrderedDict),
    # and population (IndexedOrderedDict) for faster reading
    nest_devices = Series()
    for device in ensure_list(devices):
        dev_model = device["model"]
        if dev_model in NESTInputDeviceDict.keys():
            build_device = \
                lambda nest_instance, device, populations: \
                    build_and_connect_input_device(nest_instance, device, populations)
        elif dev_model in NESTOutputDeviceDict.keys():
            build_device = \
                lambda nest_instance, device, populations: \
                    build_and_connect_output_device(nest_instance, device, populations)
        else:
            raise ValueError("Device model %s is neither one of available input devices:\n%s\n"
                             "nor of output ones!:\n%s"
                             % (dev_model, NESTInputDeviceDict.keys(), NESTOutputDeviceDict.keys()))
        device_target_nodes = device.pop("nodes", None)
        if device_target_nodes is None:
            device_target_nodes = nest_nodes
        else:
            device_target_nodes = nest_nodes[device_target_nodes]
        # Determine the connections from variables to measure/stimulate to NEST node populations
        connections = device["connections"]  # either a variable model or a dict
        if isinstance(connections, string_types):
            connections = {connections: slice(None)}  # return all population types
        # For every distinct quantity to be measured from NEST or stimulated towards NEST nodes...
        for name, populations in connections.items():
            # This set of devices will be for variable...
            nest_devices[name] = NESTDeviceSet(name, dev_model)
            # and for every target region node...
            for node in device_target_nodes:
                # and for every target node and population group...
                # create a device
                nest_devices[name][node.label] = build_device(nest_instance, device, node[populations])
    return nest_devices


def build_and_connect_output_devices(nest_instance, devices_args, nest_nodes):
    devices = build_and_connect_devices(nest_instance, devices_args, nest_nodes)
    return devices


def build_and_connect_input_devices(nest_instance, devices_args, nest_nodes):
    return build_and_connect_devices(nest_instance, devices_args, nest_nodes)

# -*- coding: utf-8 -*-
import os

import numpy as np
from pandas import Series
from six import string_types

from tvb_multiscale.core.config import CONFIGURED, initialize_logger
from tvb_multiscale.core.spiking_models.devices import DeviceSet

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.log_error_utils import raise_value_error


LOG = initialize_logger(__name__)


def log_path(name, logger=LOG):
    logger.info("%s: %s" % (name, os.environ.get(name, "")))


def _get_device_props_with_correct_shape(device, shape):

    def _assert_conn_params_shape(p, p_name, shape):
        if isinstance(p, dict):
            return np.tile(p, shape)
        elif not isinstance(p, np.ndarray):
            p = np.array(p)
        if np.any(p.shape != shape):
            if p.size == 1:
                return np.tile(p, shape)
            else:
                raise_value_error("Device %s are neither of shape (n_devices, n_nodes) = %s"
                                  "nor of size 1:\n%s" % (p_name, str(shape), str(p)))
        return p

    return _assert_conn_params_shape(device.get("weights", 1.0), "weights", shape), \
           _assert_conn_params_shape(device.get("delays", 0.0), "delays", shape), \
           _assert_conn_params_shape(device.get("receptor_type", None), "receptor_type", shape), \
           _assert_conn_params_shape(device.get("neurons_fun", None), "neurons_fun", shape)


def _get_connections(device, spiking_nodes):
    # Determine the connections from variables to measure/stimulate to Spiking node populations
    connections = device["connections"]  # either a variable model or a dict
    if isinstance(connections, string_types):
        connections = {connections: slice(None)}  # return all population types
    device_target_nodes = device.get("nodes", None)
    if device_target_nodes is None:
        device_target_nodes = spiking_nodes
    else:
        device_target_nodes = spiking_nodes[device_target_nodes]
    return connections, device_target_nodes


def build_device(device, create_device_fun, config=CONFIGURED, **kwargs):
    if isinstance(device, string_types) or isinstance(device, dict):
        if isinstance(device, string_types):
            try:
                return create_device_fun(device, config=config, **kwargs)
            except Exception as e:
                raise ValueError("Failed to set device %s!\n%s" % (str(device), str(e)))
        else:
            try:
                device_model = device.get("model", None)
                return create_device_fun(device_model, params=device.get("params", None), config=config, **kwargs)
            except Exception as e:
                raise ValueError("Failed to set device %s!\n%s" % (str(device), str(e)))
    else:
        raise ValueError("Failed to set device%s!\n "
                         "Device has to be a device model or dict!" % str(device))


def build_and_connect_device(device, create_device_fun, connect_device_fun, node, populations, inds_fun,
                             weight=1.0, delay=0.0, receptor_type=None,
                             config=CONFIGURED, **kwargs):
    device = build_device(device, create_device_fun, config=config, **kwargs)
    for pop in ensure_list(populations):
        device = connect_device_fun(device, node[pop], inds_fun,
                                    weight, delay, receptor_type, config=config, **kwargs)
    device._number_of_connections = device.number_of_connections
    return device


def build_and_connect_devices_one_to_one(device_dict, create_device_fun, connect_device_fun, spiking_nodes,
                                         config=CONFIGURED, **kwargs):
    # This function is mostly used when a measuring (output) device targets one and only Spiking node,
    # as it is the case for measuring Spiking populations from specific TVB nodes
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    devices = Series()
    # Determine the connections from variables to measure/stimulate to Spiking node populations
    connections, device_target_nodes = _get_connections(device_dict, spiking_nodes)
    # Determine the device's parameters and connections' properties
    weights, delays, receptor_types, neurons_funs = \
        _get_device_props_with_correct_shape(device_dict, (len(device_target_nodes),))
    # For every Spiking population variable to be stimulated or measured...
    for pop_var, populations in connections.items():
        # This set of devices will be for variable pop_var...
        devices[pop_var] = DeviceSet(pop_var, device_dict["model"])
        # and for every target region node...
        for i_node, node in enumerate(device_target_nodes):
            # and for every target node and population group...
            # create a device
            devices[pop_var][node.label] = \
                build_and_connect_device(device_dict, create_device_fun, connect_device_fun,
                                         node, populations, neurons_funs[i_node],
                                         weights[i_node], delays[i_node], receptor_types[i_node],
                                         config=config, **kwargs)
        devices[pop_var].update()
    return devices


def build_and_connect_devices_one_to_many(device_dict, create_device_fun, connect_device_fun, spiking_nodes,
                                          names, config=CONFIGURED, **kwargs):
    # This function is mostly used when a stimulation (input) device targets more than one Spiking node,
    # as it is the case for TVB state variables-per-node proxies
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    devices = Series()
    # Determine the connections from variables to measure/stimulate to Spiking node populations
    connections, device_target_nodes = _get_connections(device_dict, spiking_nodes)
    # Determine the device's parameters and connections' properties
    weights, delays, receptor_types, neurons_funs = \
        _get_device_props_with_correct_shape(device_dict, (len(names), len(device_target_nodes)))
    # For every Spiking population variable to be stimulated or measured...
    for pop_var, populations in connections.items():
        # This set of devices will be for variable pop_var...
        devices[pop_var] = DeviceSet(pop_var, device_dict["model"])
        # and for every target region node...
        for i_dev, dev_name in enumerate(names):
            # and for every target node and population group...
            # create a device
            devices[pop_var][dev_name] = build_device(device_dict, create_device_fun,
                                                      config=config, **kwargs)
            for i_node, node in enumerate(device_target_nodes):
                for pop in populations:
                    devices[pop_var][dev_name] = \
                       connect_device_fun(devices[pop_var][dev_name], node[pop], neurons_funs[i_dev, i_node],
                                          weights[i_dev, i_node], delays[i_dev, i_node], receptor_types[i_dev, i_node],
                                          config=config, **kwargs)
        devices[pop_var].update()
    return devices


def build_and_connect_devices(devices_input_dicts, create_device_fun, connect_device_fun, spiking_nodes,
                              config=CONFIGURED, **kwargs):
    # Build devices by their population (Series)
    # and target nodes (Series) for faster reading
    devices = Series()
    for device_dict in ensure_list(devices_input_dicts):
        # For every distinct quantity to be measured from Spiking or stimulated towards Spiking nodes...
        dev_names = device_dict.get("names", None)
        if dev_names is None:  # If no devices' names are given...
            devices = devices.append(
                            build_and_connect_devices_one_to_one(device_dict, create_device_fun, connect_device_fun,
                                                                 spiking_nodes, config=config, **kwargs)
                                              )
        else:
            devices = devices.append(
                            build_and_connect_devices_one_to_many(device_dict, create_device_fun, connect_device_fun,
                                                                  spiking_nodes, dev_names, config=config, **kwargs)
                                              )
    return devices

# -*- coding: utf-8 -*-

import numpy as np
from six import string_types
from copy import deepcopy

from tvb_multiscale.core.config import CONFIGURED, initialize_logger
from tvb_multiscale.core.spiking_models.devices import DeviceSet, DeviceSets

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.log_error_utils import raise_value_error


LOG = initialize_logger(__name__)


def _safe_indexing(arg, ind1=None, ind2=None):
    if ind1 is None:
        return deepcopy(arg)
    else:
        if ind2 is None:
            try:
                return deepcopy(arg[ind1])
            except:
                return deepcopy(arg)
        else:
            try:
                return deepcopy(arg[ind1, ind2])
            except:
                return deepcopy(arg)


def _unload_device_dict(device):
    return device.pop("weights", 1.0), \
           device.pop("delays", 0.0), \
           device.pop("receptor_type", None), \
           device.pop("syn_spec", None), \
           device.pop("conn_spec", None), \
           device.pop("neurons_fun", None)


def _get_connections(device, spiking_nodes):
    # Determine the connections
    # from variables to measure/stimulate
    # to Spiking node populations
    connections = device["connections"]
    if isinstance(connections, string_types):
        connections = {connections: slice(None)}  # return all population types
    device_target_nodes = device.get("nodes", None)
    if device_target_nodes is None:
        device_target_nodes = spiking_nodes
    else:
        device_target_nodes = spiking_nodes[device_target_nodes]
    return connections, device_target_nodes


def build_device(device, create_device_fun, config=CONFIGURED, **kwargs):
    """This method will only build a device based on the input create_device_fun function,
       which is specific to every spiking simulator.
       Arguments:
        device: either a name (string) of a device model, or a dictionary of properties for the device to build
        create_device_fun: a function to build the device
        config: a configuration class instance. Default = CONFIGURED (default configuration)
        **kwargs: other possible keyword arguments, to be passed to the device builder.
       Returns:
        the built Device class instance
    """
    if isinstance(device, string_types) or isinstance(device, dict):
        if isinstance(device, string_types):
            try:
                return create_device_fun(device, config=config, **kwargs)
            except Exception as e:
                raise ValueError("Failed to set device %s!\n%s" % (str(device), str(e)))
        else:
            try:
                device_model = device.get("model", None)
                return create_device_fun(device_model, params=deepcopy(device.get("params", dict())),
                                         config=config, **kwargs)
            except Exception as e:
                raise ValueError("Failed to set device %s!\n%s" % (str(device), str(e)))
    else:
        raise ValueError("Failed to set device%s!\n "
                         "Device has to be a device model or dict!" % str(device))


def build_and_connect_device(device, create_device_fun, connect_device_fun, node, populations, inds_fun,
                             weight=1.0, delay=0.0, receptor_type=None, syn_spec=None, conn_spec=None,
                             config=CONFIGURED, **kwargs):
    """This method will build a device and connect it to the spiking network
       based on the input create_device_fun, and connect_device_fun functions,
       which are specific to every spiking simulator.
       Arguments:
        device: either a name (string) of a device model, or a dictionary of properties for the device to build
        create_device_fun: a function to build the device
        connect_device_fun: a function to connect the device
        node: the target SpikingRegionNode class instance
        populations: target populations' labels:
        inds_fun: a function to select a subset of each population's neurons
        weight: the weight of the connection. Default = 1.0
        delay: the delay of the connection. Default = 0.0
        receptor_type: the synaptic receptor type of the connection. Default = None,
                       which will default in a way specific to each spiking simulator.
        config: a configuration class instance. Default = CONFIGURED (default configuration)
        **kwargs: other possible keyword arguments, to be passed to the device builder.
       Returns:
        the built and connected Device class instance
    """
    device = build_device(device, create_device_fun, config=config, **kwargs)
    for pop in ensure_list(populations):
        device = connect_device_fun(device, node[pop], inds_fun,
                                    weight, delay, receptor_type, syn_spec, conn_spec, config=config, **kwargs)
    return device


def build_and_connect_devices_one_to_one(device_dict, create_device_fun, connect_device_fun, spiking_nodes,
                                         config=CONFIGURED, **kwargs):
    """This function will create a DeviceSet for a measuring (output) or input (stimulating) quantity,
       whereby each device will target one and only SpikingRegionNode,
       e.g. as it is the case for measuring Spiking populations from specific TVB nodes."""
    devices = kwargs.get("devices", DeviceSets())
    # Determine the connections from variables to measure/stimulate to Spiking node populations
    connections, device_target_nodes = _get_connections(device_dict, spiking_nodes)
    # Determine the device's parameters and connections' properties
    weights, delays, receptor_types, syn_specs, conn_specs, neurons_funs = _unload_device_dict(device_dict)
    # For every Spiking population variable to be stimulated or measured...
    for dev_var, populations in connections.items():
        populations = ensure_list(populations)
        # This set of devices will be for variable pop_var...
        if dev_var not in devices.keys():
            devices[dev_var] = DeviceSet(name=dev_var, model=device_dict["model"])
        # and for every target region node...
        for i_node, node in enumerate(device_target_nodes):
            # ...and populations' group...
            # ...create a device and connect it:
            kwargs.update({"label": "%s - %s" % (dev_var, node.label)})
            if node.label not in devices[dev_var].keys():
                devices[dev_var][node.label] = build_device(device_dict, create_device_fun, config=config, **kwargs)
            for pop in populations:
                if isinstance(pop, np.ndarray):
                    population = pop.item()
                else:
                    population = pop
                devices[dev_var][node.label] = \
                    connect_device_fun(devices[dev_var][node.label], node[population],
                                       _safe_indexing(neurons_funs, i_node),
                                       _safe_indexing(weights, i_node),
                                       _safe_indexing(delays, i_node),
                                       _safe_indexing(receptor_types, i_node),
                                       _safe_indexing(syn_specs, i_node),
                                       _safe_indexing(conn_specs, i_node),
                                       config=config, **kwargs)
        devices[dev_var].update()
    return devices


def build_and_connect_devices_one_to_many(device_dict, create_device_fun, connect_device_fun, spiking_nodes,
                                          names, config=CONFIGURED, **kwargs):
    """This function will create a DeviceSet for a measuring (output) or input (stimulating) quantity,
       whereby each device will target more than one SpikingRegionNode instances,
       e.g. as it is the case a TVB "proxy" node,
       stimulating several of the SpikingRegionNodes in the spiking network."""
    devices = kwargs.get("devices", DeviceSets())
    # Determine the connections from variables to measure/stimulate to Spiking node populations
    connections, device_target_nodes = _get_connections(device_dict, spiking_nodes)
    # Determine the device's parameters and connections' properties
    weights, delays, receptor_types, syn_specs, conn_specs, neurons_funs = _unload_device_dict(device_dict)
    # For every Spiking population variable to be stimulated or measured...
    for pop_var, populations in connections.items():
        populations = ensure_list(populations)
        # This set of devices will be for variable pop_var...
        if pop_var not in devices.keys():
            devices[pop_var] = DeviceSet(name=pop_var, model=device_dict["model"])
        # and for every target region node...
        for i_dev, dev_name in enumerate(names):
            # ...and populations' group...
            # create a device
            if dev_name not in devices[pop_var].keys():
                devices[pop_var][dev_name] = build_device(device_dict, create_device_fun, config=config,
                                                          label="%s - %s" % (pop_var, dev_name), **kwargs)
            # ...and loop through the target region nodes...
            for i_node, node in enumerate(device_target_nodes):
                # ...and populations' group...
                # ...to connect it:
                for pop in populations:
                    if isinstance(pop, np.ndarray):
                        population = pop.item()
                    else:
                        population = pop
                    devices[pop_var][dev_name] = \
                        connect_device_fun(devices[pop_var][dev_name], node[population],
                                           _safe_indexing(neurons_funs, i_dev, i_node),
                                           _safe_indexing(weights, i_dev, i_node),
                                           _safe_indexing(delays, i_dev, i_node),
                                           _safe_indexing(receptor_types, i_dev, i_node),
                                           _safe_indexing(syn_specs, i_dev, i_node),
                                           _safe_indexing(conn_specs, i_dev, i_node),
                                           config=config, **kwargs)
        devices[pop_var].update()
    return devices


def build_and_connect_devices(devices_input_dicts, create_device_fun, connect_device_fun, spiking_nodes,
                              config=CONFIGURED, **kwargs):
    """A method to build and connect devices by the variable they measure or stimulate,
       and population(s) they target and target node (where they refer to.
    """
    devices = kwargs.pop("devices", DeviceSets())
    for device_dict in ensure_list(devices_input_dicts):
        # For every distinct quantity to be measured from Spiking or stimulated towards Spiking nodes...
        dev_names = device_dict.get("names", None)
        if dev_names is None:  # If no devices' names are given...
            devices = build_and_connect_devices_one_to_one(device_dict, create_device_fun, connect_device_fun,
                                                           spiking_nodes, devices=devices, config=config, **kwargs)

        else:
            devices = build_and_connect_devices_one_to_many(device_dict, create_device_fun, connect_device_fun,
                                                            spiking_nodes, dev_names, devices=devices, config=config,
                                                            **kwargs)
    return devices

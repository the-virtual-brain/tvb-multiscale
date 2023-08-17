# -*- coding: utf-8 -*-

"""
Functions and/or templates used to set parameters of populations, devices and their connections

Template used to set neural population parameters:
Functions are accepted only for:
"params": the parameters of the spiking neuron
"scale": the float scaling the population_order to get the size of the population
Such functions will depend only on the brain region "node" argument.
Set different entries in the SpikingNetwork builder for populations that differ in
"label", "model" or "nodes"
"""


def set_neural_population_params(node):
    return {}
    # return dict


def set_neural_population_scale(node):
    return 1.0
    # return float


"""" Template used to set neural population connection parameters within region node:
Functions are accepted only for:
"weight": the weights of the connections between neurons of the source and target populations
"delay": the delays of the connections between neurons of the source and target populations
"receptor_type": the receptor_type of the connections between neurons of the source and target populations
Such functions will depend only on the brain region "node" argument.
Set different entries in the SpikingNetwork builder for populations that differ in
"source": the source population label
"target": the target population label or list of labels"model" or "nodes"
"model": the model of the synapse
"conn_spec": the dictionary of the connectivity model and parameters (See NEST)
or their nodes ranges
"""


def set_within_node_connection_weight(node):
    return 1.0
    # return float or dict


def set_within_node_connection_delay(node):
    return 1.0
    # return float or dict


"""
Template used to set neural population connection parameters between distinct region node:
Functions are accepted only for:
"weight": the weights of the connections between neurons of the source and target populations
"delay": the delays of the connections between neurons of the source and target populations
"receptor_type": the receptor_type of the connections between neurons of the source and target populations
Such functions will depend only on the brain regions "source_node", "target_node" arguments,
and on tvb connectivity (see below)
Set different entries in the SpikingNetwork builder for populations that differ in
"source": the source population label
"target": the target population label or list of labels"model" or "nodes"
"model": the model of the synapse
"conn_spec": the dictionary of the connectivity model and parameters (See NEST)
or in their source_nodes and target_nodes ranges
"""


def set_between_nodes_connection_weight(source_node, target_node, tvb_weights):
    return 1.0
    # return float or dict


def tvb_weight(source_node, target_node, tvb_weights):
    return tvb_weights[target_node, source_node].item()


def scale_tvb_weight(source_node, target_node, tvb_weights, scale=1.0):
    return scale * tvb_weights[target_node, source_node].item()


def set_between_nodes_connection_delay(source_node, target_node, tvb_delays):
    return 1.0
    # return float or dict


def tvb_delay(source_node, target_node, tvb_delays):
    return tvb_delays[target_node, source_node].item()


def scale_tvb_delay(source_node, target_node, tvb_delays, scale=1.0):
    return (scale * tvb_delays[target_node, source_node]).item()


def add_to_tvb_delay(source_node, target_node, tvb_delays, add=1.0):
    return (add + tvb_delays[target_node, source_node]).item()


"""
Template used to set input/stimulation devices parameters towards region nodes:
Functions are accepted only for:
"params": device parameters
"weight": the weights of the connections towards neurons of the target populations
"delay": the delays of the connections towards neurons of the target populations
"receptor_type": the receptor_type of the connections towards neurons of the target populations
Such functions will depend only on the target region "node" argument.
Set different entries in the SpikingNetwork builder for devices that differ in
"connections": connections dict between device label and target population
"model": the model of the synapse
or their target nodes ranges
"""


def set_input_device_params(node):
    return {}
    # return dict


def set_input_device_connections(node):
    return {}
    # return dict


def set_input_device_weight(node):
    return 1.0
    # return float or dict


def set_input_device_connection_delay(node):
    return 1.0
    # return float or dict


def set_input_device_connection_receptor_type(node):
    return int(0)


"""
Template used to set output/measuring devices parameters towards region nodes:
Functions are accepted only for:
"params": device parameters
"weight": the weights of the connections towards neurons of the target populations
"delay": the delays of the connections towards neurons of the target populations
Such functions will depend only on the source region "node" argument.
Set different entries in the SpikingNetwork builder for devices that differ in
"connections": connections dict between device label and target population
"model": the model of the synapse
or their source nodes ranges
"""


def set_output_device_params(node):
    return {}

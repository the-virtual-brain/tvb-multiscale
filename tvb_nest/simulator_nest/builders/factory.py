# -*- coding: utf-8 -*-

# Functions and/or templates used to set parameters of populations and/or connections


# Template used to set neural population parameters:
# Functions are accepted only for
# "params": the parameters of the spiking neuron
# "scale": the float scaling the population_order to get the size of the population
# Such functions will depend only on the brain region "node" argument.
# Set different entries in the NESTNetwork builder for populations that differ in
# "label", "model" or "nodes"

def set_neural_population_params(node):
    return {}
    # return dict


def set_neural_population_scale(node):
    return 1.0
    # return float


# Template used to set neural population connection parameters within region node:
# Functions are accepted only for
# "weight": the weights of the connections between neurons of the source and target populations
# "delay": the delays of the connections between neurons of the source and target populations
# "receptor_type": the receptor_type of the connections between neurons of the source and target populations
# Such functions will depend only on the brain region "node" argument.
# Set different entries in the NESTNetwork builder for populations that differ in
# "source": the source population label
# "target": the target population label or list of labels"model" or "nodes"
# "model": the model of the synapse
# "conn_spec": the dictionary of the connectivity model and parameters (See NEST)
def set_within_node_connection_weight(node):
    return 1.0
    # return float or dict


def set_within_node_connection_random_normal_weight(node, mu=1.0, sigma=0.1):
    return {"distribution": "normal", "mu": mu, "sigma": sigma}
    # return float or dict


def set_within_node_connection_delay(node):
    return 0.0
    # return float or dict


def set_within_node_connection_receptor_type(node):
    return int(0)
    # return int


# Template used to set neural population connection parameters within region node:
# Functions are accepted only for
# "weight": the weights of the connections between neurons of the source and target populations
# "delay": the delays of the connections between neurons of the source and target populations
# "receptor_type": the receptor_type of the connections between neurons of the source and target populations
# Such functions will depend only on the brain regions "source_node", "target_node" arguments,
# and on tvb connectivity (see below)
# Set different entries in the NESTNetwork builder for populations that differ in
# "source": the source population label
# "target": the target population label or list of labels"model" or "nodes"
# "model": the model of the synapse
# "conn_spec": the dictionary of the connectivity model and parameters (See NEST)
def set_between_nodes_connection_weight(source_node, target_node, tvb_weights):
    return 1.0
    # return float or dict


def tvb_weight(source_node, target_node, tvb_weights):
    return tvb_weights[source_node, target_node]


def scale_tvb_weight(source_node, target_node, tvb_weights, scale=1.0):
    return scale * tvb_weights[source_node, target_node]


def random_normal_tvb_weight(source_node, target_node, tvb_weights, sigma=0.1):
    return {"distribution": "normal",
            "mu": tvb_weights[source_node, target_node],
            "sigma": sigma*tvb_weights[source_node, target_node]}


def set_between_nodes_connection_delay(source_node, target_node, tvb_delays):
    return 0.0
    # return float or dict


def tvb_delay(source_node, target_node, tvb_delays):
    return tvb_delays[source_node, target_node]


def scale_tvb_delay(source_node, target_node, tvb_delays, scale=1.0):
    return scale * tvb_delays[source_node, target_node]


def add_to_tvb_delay(source_node, target_node, tvb_delays, add=1.0):
    return add + tvb_delays[source_node, target_node]


def set_between_nodes_connection_receptor_type(source_node, target_node):
    return int(0)
    # return int


def receptor_by_source_region(source_node, target_node, start=1):
    return int(start + source_node)

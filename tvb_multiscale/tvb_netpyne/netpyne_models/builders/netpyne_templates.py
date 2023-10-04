# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.spiking_models.builders.templates import tvb_weight, tvb_delay

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

# def set_within_node_connection_random_normal_weight(node, mu=1.0, sigma=0.1):
#     return {"distribution": "normal", "mu": mu, "sigma": sigma}
#     # return float or dict


# def set_within_node_connection_receptor_type(node):
#     return int(0)
#     # return int

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

def random_normal_weight(weight, scale=1.0, sigma=0.1):
    w = weight * scale
    # TODO: in NEURON, it's variance, not sigma!
    return f"max(0.0, normal({w}, {w * sigma}))"

def random_normal_tvb_weight(source_node, target_node, tvb_weights, scale=1.0, sigma=0.1):
    return random_normal_weight(tvb_weight(source_node, target_node, tvb_weights), scale, sigma)

def random_uniform_delay(delay, low=0.0, high=1.0, sigma=0.1):
    if delay == 0:
        return delay
    if sigma is not None and sigma > 0.0:
        low = np.maximum(low, (1-sigma) * delay)
        high = np.maximum(high, (1 + sigma) * delay)
    if high <= low:
        raise ValueError("Maximum delay %f is not larger than the minimum one %f!" % (high, low))
    return f"uniform({low}, {high})"

def random_uniform_tvb_delay(source_node, target_node, tvb_delays, low=0.0, high=1.0, sigma=0.1):
    return random_uniform_delay(tvb_delay(source_node, target_node, tvb_delays), low, high, sigma)

def receptor_by_source_region(source_node, target_node, start=1):
    return int(start + source_node)
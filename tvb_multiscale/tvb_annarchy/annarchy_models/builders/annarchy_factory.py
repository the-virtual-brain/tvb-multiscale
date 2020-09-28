# -*- coding: utf-8 -*-

import os
import sys
import shutil
from six import string_types
import numpy as np

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import ANNARCHYInputDeviceDict, ANNARCHYOutputDeviceDict
from tvb_multiscale.core.spiking_models.builders.factory import log_path

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error, warning
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs, delete_folder_safely


LOG = initialize_logger(__name__)


#TODO: Find a better way to abstract between nest_factory and factory!


# Helper functions with NEST


def load_annarchy(config=CONFIGURED, logger=LOG, clean_compilation_directory=True, **kwargs):
    logger.info("Loading an ANNarchy instance...")
    if clean_compilation_directory:
        logger.info("Cleaning ANNarchy compilation directory, if any...")
        delete_folder_safely(os.path.join(os.getcwd(), "annarchy"))
    import ANNarchy as annarchy_instance
    annarchy_instance.clear()
    if len(kwargs):
        logger.info("Configuring ANNarchy with properties:\n%s" % str(kwargs))
        annarchy_instance.setup(**kwargs)
    return annarchy_instance


def compile(modules, recompile=False, config=CONFIGURED, logger=LOG):
    pass


def create_conn_spec(n_src=1, n_trg=1, src_is_trg=False, config=CONFIGURED, **kwargs):
    # This function returns a conn_spec dictionary
    # and the expected/accurate number of total connections
    conn_spec = dict(config.DEFAULT_CONNECTION["conn_spec"])
    P_DEF = conn_spec["p"]
    conn_spec.update(kwargs)
    rule = conn_spec["rule"]
    p = conn_spec["p"]
    N = conn_spec["N"]
    autapses = conn_spec["allow_autapses"]
    multapses = conn_spec["allow_multapses"]
    indegree = conn_spec["indegree"]
    outdegree = conn_spec["outdegree"]
    conn_spec = {
        'rule': rule,
        'allow_autapses': autapses,  # self-connections flag
        'allow_multapses': multapses  # multiple connections per neurons' pairs flag
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


def connect(source_pop, target_pop, target, synapse=None, name=None,
            method="connect_all_to_all", annarchy_instance=None, **connection_args):
    """
    Method to set up and connect a projection between two ANNarchyPopulations.
    Arguments:
      - target_pop: The ANNarchyPopulation we want to connect to.
      - target: type of the connection. Needs to be set, or weights are zero.
      - synapse: a ``Synapse`` instance.
      - name: name of the projection
      - method: name of an ANNarchy connection method
      - **connection_args: depend on the chosen ANNarchy connection method
      Returns: the projection
      """
    proj = annarchy_instance.Projection(source_pop, target_pop, target, synapse, name)
    source_pop._projections_pre.append(proj)  # is there a faster way than using .append()?
    target_pop._projections_post.append(proj)
    return getattr(proj, method)(**connection_args)


def device_to_dev_model(device):
    if device == "spike_multimeter":
        return "multimeter"
    else:
        return device


def create_device(device_model, device_name=None, params=None, config=CONFIGURED, annarchy_instance=None):
    if annarchy_instance is None:
        annarchy_instance = load_annarchy(config=config)
        return_annarchy = True
    else:
        return_annarchy = False
    if not isinstance(device_name, string_types):
        device_name = device_model
    else:
        # ...assert the type...
        device_model = device_to_dev_model(device_name)
    if device_model in ANNARCHYInputDeviceDict.keys():
        devices_dict = ANNARCHYInputDeviceDict
        default_params_dict = config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF
    elif device_model in ANNARCHYOutputDeviceDict.keys():
        devices_dict = ANNARCHYOutputDeviceDict
        default_params_dict = config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF),
                           str(config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF)))
    default_params = dict(default_params_dict.get(device_name, {}))
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
    # TODO: a better solution for the strange error with inhomogeneous poisson generator
    try:
        annarchy_device_id = annarchy_instance.Create(device_model, params=default_params)
    except:
        warning("Using temporary hack for creating successive %s devices!" % device_model)
        annarchy_device_id = annarchy_instance.Create(device_model, params=default_params)
    annarchy_device = devices_dict[device_name](annarchy_device_id, annarchy_instance)
    if return_annarchy:
        return annarchy_device, annarchy_instance
    else:
        return annarchy_device


def connect_device(nest_device, neurons, weight=1.0, delay=0.0, receptor_type=0, config=CONFIGURED,
                   nest_instance=None):
    if nest_instance is None:
        raise_value_error("There is no NEST instance!")
    resolution = nest_instance.GetKernelStatus("resolution")
    if isinstance(delay, dict):
        if delay["low"] < resolution:
            delay["low"] = resolution
            warning("Minimum delay %f is smaller than the NEST simulation resolution %f!\n"
                    "Setting minimum delay equal to resolution!" % (delay["low"], resolution))
        if delay["high"] <= delay["low"]:
            raise_value_error("Maximum delay %f is not smaller than minimum one %f!" % (delay["high"], delay["low"]))
    else:
        if delay < resolution:
            delay = resolution
            warning("Delay %f is smaller than the NEST simulation resolution %f!\n"
                    "Setting minimum delay equal to resolution!" % (delay, resolution))
    syn_spec = {"weight": weight, "delay": delay, "receptor_type": receptor_type}
    if nest_device.model == "spike_detector":
        #                     source  ->  target
        nest_instance.Connect(neurons, nest_device.device, syn_spec=syn_spec)
    else:
        nest_instance.Connect(nest_device.device, neurons, syn_spec=syn_spec)
    # nest_device.update_number_of_connections()
    return nest_device
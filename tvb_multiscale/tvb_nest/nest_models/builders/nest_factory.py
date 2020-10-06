# -*- coding: utf-8 -*-

import os
import sys
import shutil
from six import string_types
import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.devices import NESTInputDeviceDict, NESTOutputDeviceDict
from tvb_multiscale.core.spiking_models.builders.factory import log_path

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error, warning
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs


LOG = initialize_logger(__name__)


#TODO: Find a better way to abstract between nest_factory and factory!


# Helper functions with NEST


def load_nest(config=CONFIGURED, logger=LOG):

    logger.info("Loading a NEST instance...")
    nest_path = config.NEST_PATH
    os.environ['NEST_INSTALL_DIR'] = nest_path
    log_path('NEST_INSTALL_DIR', logger)
    os.environ['NEST_DATA_DIR'] = os.path.join(nest_path, "share/nest")
    log_path('NEST_DATA_DIR', logger)
    os.environ['NEST_DOC_DIR'] = os.path.join(nest_path, "share/doc/nest")
    log_path('NEST_DOC_DIR', logger)
    os.environ['NEST_MODULE_PATH'] = os.path.join(nest_path, "lib/nest")
    log_path('NEST_MODULE_PATH', logger)
    os.environ['PATH'] = os.path.join(nest_path, "bin") + ":" + os.environ['PATH']
    log_path('PATH', logger)
    LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', '')
    if len(LD_LIBRARY_PATH) > 0:
        LD_LIBRARY_PATH = ":" + LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = os.environ['NEST_MODULE_PATH'] + LD_LIBRARY_PATH
    log_path('LD_LIBRARY_PATH', logger)
    os.environ['SLI_PATH'] = os.path.join(os.environ['NEST_DATA_DIR'], "sli")
    log_path('SLI_PATH', logger)

    os.environ['NEST_PYTHON_PREFIX'] = config.PYTHON
    log_path('NEST_PYTHON_PREFIX', logger)
    sys.path.insert(0, os.environ['NEST_PYTHON_PREFIX'])
    logger.info("%s: %s" % ("system path", sys.path))

    import nest
    return nest


def compile_modules(modules, recompile=False, config=CONFIGURED, logger=LOG):
    # ...unless we need to first compile it:
    from pynestml.frontend.pynestml_frontend import install_nest
    logger.info("Preparing MYMODULES_BLD_DIR: %s" % config.MYMODULES_BLD_DIR)
    safe_makedirs(config.MYMODULES_BLD_DIR)
    for module in ensure_list(modules):
        logger.info("Compiling %s..." % module)
        module_bld_dir = os.path.join(config.MYMODULES_BLD_DIR, module)
        logger.info("in build directory %s..." % module_bld_dir)
        if not os.path.exists(module_bld_dir) or recompile:
            source_path = os.path.join(config.MYMODULES_DIR, module)
            logger.info("copying sources from %s\ninto %s..." % (source_path, module_bld_dir))
            shutil.copytree(source_path, module_bld_dir)
        logger.info("Running compilation...")
        install_nest(module_bld_dir, config.NEST_PATH)
        if os.path.isfile(os.path.join(config.MYMODULES_BLD_DIR, "lib" + module + "module.so")):
            logger.info("DONE compiling %s!" % module)
        else:
            logger.warn("Something seems to have gone wrong with compiling %s!" % module)


def get_populations_neurons(population, inds_fun=None):
    if inds_fun is None:
        return population._population
    return inds_fun(population._population)


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


def device_to_dev_model(device):
    if device == "spike_multimeter":
        return "multimeter"
    else:
        return device


def create_device(device_model, params=None, config=CONFIGURED, nest_instance=None):
    if nest_instance is None:
        nest_instance = load_nest(config=config)
        return_nest = True
    else:
        return_nest = False
    # Assert the model name...
    device_model = device_to_dev_model(device_model)
    if device_model in NESTInputDeviceDict.keys():
        devices_dict = NESTInputDeviceDict
        default_params_dict = config.NEST_INPUT_DEVICES_PARAMS_DEF
    elif device_model in NESTOutputDeviceDict.keys():
        devices_dict = NESTOutputDeviceDict
        default_params_dict = config.NEST_OUTPUT_DEVICES_PARAMS_DEF
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.NEST_INPUT_DEVICES_PARAMS_DEF),
                           str(config.NEST_OUTPUT_DEVICES_PARAMS_DEF)))
    default_params = dict(default_params_dict.get(device_model, {}))
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
    # TODO: a better solution for the strange error with inhomogeneous poisson generator
    label = default_params.pop("label", "")
    try:
        nest_device_id = nest_instance.Create(device_model, params=default_params)
    except:
        warning("Using temporary hack for creating successive %s devices!" % device_model)
        nest_device_id = nest_instance.Create(device_model, params=default_params)
    nest_device = devices_dict[device_model](nest_device_id, nest_instance, label=label)
    if return_nest:
        return nest_device, nest_instance
    else:
        return nest_device


def connect_device(nest_device, population, neurons_inds_fun, weight=1.0, delay=0.0, receptor_type=0,
                   nest_instance=None, config=CONFIGURED):
    if receptor_type is None:
        receptor_type = 0
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
    neurons = get_populations_neurons(population, neurons_inds_fun)
    if nest_device.model == "spike_detector":
        #                     source  ->  target
        nest_instance.Connect(neurons, nest_device.device, syn_spec=syn_spec)
    else:
        nest_instance.Connect(nest_device.device, neurons, syn_spec=syn_spec)
    return nest_device

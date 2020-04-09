# -*- coding: utf-8 -*-

import os
import shutil
import sys

import numpy as np
from six import string_types
from tvb.simulator.plot.utils.data_structures_utils import ensure_list
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error

from tvb_multiscale.spiking_models.builders.factory import log_path
from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.devices import NESTInputDeviceDict, NESTOutputDeviceDict

LOG = initialize_logger(__name__)


# TODO: Find a better way to abstract between nest_factory and factory!


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
    if not os.path.exists(config.MODULES_BLDS_DIR):
        logger.info("Creating MODULES_BLDS_DIR: %s" % config.MODULES_BLDS_DIR)
        os.makedirs(config.MODULES_BLDS_DIR)
    for module in ensure_list(modules):
        logger.info("Compiling %s..." % module)
        module_bld_dir = os.path.join(config.MODULES_BLDS_DIR, module)
        logger.info("from in build_interfaces directory %s..." % module_bld_dir)
        if not os.path.exists(module_bld_dir) or recompile:
            source_path = os.path.join(config.MODULES_DIR, module)
            logger.info("copying sources from %s\ninto %s..." % (source_path, module_bld_dir))
            shutil.copytree(source_path, module_bld_dir)
        logger.info("Running compilation...")
        install_nest(module_bld_dir, config.NEST_PATH)
        if os.path.isfile(os.path.join(config.MODULES_BLDS_DIR, module + "module.so")) and \
                os.path.isfile(os.path.join(config.MODULES_BLDS_DIR, "lib" + module + "module.so")):
            logger.info("DONE compiling %s!" % module)
        else:
            logger.warn("Something seems to have gone wrong with compiling %s!" % module)


def create_conn_spec(n_src=1, n_trg=1, src_is_trg=False, config=CONFIGURED, **kwargs):
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


def device_to_dev_model(device):
    if device == "spike_multimeter":
        return "multimeter"
    else:
        return device


def create_device(device_model, device_name=None, params=None, config=CONFIGURED, nest_instance=None):
    if nest_instance is None:
        nest_instance = load_nest(config=config)
        return_nest = True
    else:
        return_nest = False
    if not isinstance(device_name, string_types):
        device_name = device_model
    else:
        # ...assert the type...
        device_model = device_to_dev_model(device_name)
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
    default_params = dict(default_params_dict.get(device_name, {}))
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
    if return_nest:
        return devices_dict[device_name](nest_instance.Create(device_model, params=default_params), nest_instance), \
               nest_instance
    else:
        return devices_dict[device_name](nest_instance.Create(device_model, params=default_params), nest_instance)


def connect_device(nest_device, neurons, weight=1.0, delay=0.0, receptor_type=0, config=CONFIGURED,
                   nest_instance=None):
    if nest_instance is None:
        raise_value_error("There is no NEST instance!")
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

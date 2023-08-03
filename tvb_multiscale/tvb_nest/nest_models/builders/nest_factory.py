# -*- coding: utf-8 -*-

import os
import shutil
from copy import deepcopy

import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED, initialize_logger

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error, warning
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs


LOG = initialize_logger(__name__)


#TODO: Find a better way to abstract between nest_factory and factory!


# Helper functions with NEST


def load_nest(config=CONFIGURED, logger=LOG):
    """This method will load a NEST instance and return it, after reading the NEST environment constants.
        Arguments:
         config: configuration class instance. Default: imported default CONFIGURED object.
         logger: logger object. Default: local LOG object.
        Returns:
         the imported NEST instance
    """
    try:
        import nest
    except:
        config.configure_nest_path(logger=logger)
        import nest

    nest.ResetKernel()
    return nest


def configure_nest_kernel(nest_instance, config=CONFIGURED):
    nest_instance.ResetKernel()  # This will restart NEST!
    nest_instance.set_verbosity(config.NEST_VERBOCITY)  # don't print all messages from NEST
    # Printing the time progress should only be used when the simulation is run on a local machine:
    #  kernel_config["print_time"] = self.nest_instance.Rank() == 0
    kernel_config = deepcopy(config.DEFAULT_NEST_KERNEL_CONFIG)
    kernel_config["data_path"] = os.path.join(config.out.FOLDER_RES, os.path.basename(kernel_config["data_path"]))
    safe_makedirs(kernel_config["data_path"])  # Make sure this folder exists
    nest_instance.SetKernelStatus(kernel_config)
    return nest_instance

    
def compile_modules(modules, recompile=False, config=CONFIGURED, logger=LOG):
    """Function to compile NEST modules.
       Arguments:
        modules: a sequence (list, tuple) of NEST modules' names (strings).
        recompile: (bool) flag to recompile a module that is already compiled. Default = False.
        config: configuration class instance. Default: imported default CONFIGURED object.
        logger: logger object. Default: local LOG object.
    """
    # ...unless we need to first compile it:
    logger.info("Preparing MYMODULES_BLD_DIR: %s" % config.MYMODULES_BLD_DIR)
    safe_makedirs(config.MYMODULES_BLD_DIR)
    # TODO: find out these locations for the newest versions of NESTML and NEST!!!
    # lib_path = os.path.join(config._NEST_PATH, "lib", "nest")
    # include_path = os.path.join(config._NEST_PATH, "include")
    for module in ensure_list(modules):
        modulemodule = module + "module"
        module_bld_dir = os.path.join(config.MYMODULES_BLD_DIR, module)
        # solib_file = os.path.join(module_bld_dir, modulemodule + ".so")
        # dylib_file = os.path.join(module_bld_dir, "lib" + modulemodule + ".dylib")
        # include_file = os.path.join(module_bld_dir, modulemodule + ".h")
        # installed_solib_file = os.path.join(lib_path, os.path.basename(solib_file))
        # installed_dylib_file = os.path.join(lib_path, os.path.basename(dylib_file))
        module_include_path = os.path.join(include_path, modulemodule)
        installed_h_file = os.path.join(module_include_path, modulemodule + ".h")
        # if self.NEST_PATH and \
        #     (not os.path.isfile(solib_file)
        #         or not os.path.isfile(dylib_file)
        #             or not os.path.isfile(include_file)) \
        #                 or recompile:
        #     # If any of the .so, .dylib or .h files don't exist,
        #     # or if the user requires recompilation,
        #     # proceed with recompilation:
        if os.path.exists(module_bld_dir):
            # Delete any pre-compiled built files:
            shutil.rmtree(module_bld_dir)
        # Create a  module build directory and copy there the source files:
        source_path = os.path.join(config.MYMODULES_DIR, module)
        logger.info("Copying module sources from %s\ninto %s..." % (source_path, module_bld_dir))
        shutil.copytree(source_path, module_bld_dir)
        # Now compile:
        logger.info("Compiling %s..." % module)
        logger.info("in build directory %s..." % module_bld_dir)
        success_message = "DONE compiling and installing %s!" % module
        try:
            from pynestml.frontend.pynestml_frontend import generate_nest_target
            generate_nest_target(module_bld_dir, config._NEST_PATH)
        except Exception as e:
            raise e
        logger.info("Compiling finished without errors...")
        # else:
            # logger.info("Installing precompiled module %s..." % module)
            # success_message = "DONE installing precompiled module %s!" % module
            # # Just copy the .h, .so, and .dylib files to the appropriate NEST build paths:
            # shutil.copyfile(solib_file, installed_solib_file)
            # shutil.copyfile(solib_file, installed_dylib_file)
            # safe_makedirs(include_path)
            # shutil.copyfile(os.path.join(module_bld_dir, modulemodule + ".h"), installed_h_file)
        # installed_files = {}
        # for file in [installed_solib_file, installed_dylib_file, installed_h_file]:
        #     installed_files[file] = os.path.isfile(file)
        # if all(installed_files.values()):
        logger.info(success_message)
        # else:
        #     logger.warn("Something seems to have gone wrong with compiling and/or installing %s!"
        #                 "\n Installed files (not) found (True (False) respectively)!:\n%s"
        #                 % (module, str(installed_files)))


def get_populations_neurons(population, inds_fun=None):
    """This method will return a subset NEST.NodeCollection instance
       of the NESTPopulation._population, if inds_fun argument is a function
       Arguments:
        population: a NESTPopulation class instance
        inds_fun: a function that takes a NEST.NodeCollection as argument and returns another NEST.NodeCollection
       Returns:
        NEST.NodeCollection NESTPopulation._population instance
    """
    if inds_fun is None:
        return population._nodes
    return inds_fun(population._nodes)


def create_conn_spec(n_src=1, n_trg=1, src_is_trg=False, config=CONFIGURED, **kwargs):
    """This function returns a conn_spec dictionary and the expected/accurate number of total connections.
       Arguments:
        n_src: number (int) of source neurons. Default = 1.
        n_trg: number (int) of target neurons. Default = 1.
        src_is_trg: a (bool) flag to determine if the source and target populations are the same one. Default = False.
        config: configuration class instance. Default: imported default CONFIGURED object.
    """
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
    """Method to return a multimeter device for a spike_multimeter model name."""
    if device == "spike_multimeter":
        return "multimeter"
    elif device.find("parrot") > -1:
        return device.split("parrot_")[-1]
    else:
        return device


def create_device(device_model, params=dict(), config=CONFIGURED, nest_instance=None, **kwargs):
    """Method to create a NESTDevice.
       Arguments:
        device_model: name (string) of the device model
        params: dictionary of parameters of device and/or its synapse. Default = {}
        config: configuration class instance. Default: imported default CONFIGURED object.
        nest_instance: the NEST instance.
                       Default = None, in which case we are going to load one, and also return it in the output
       Returns:
        the NESTDevice class, and optionally, the NEST instance if it is loaded here.
    """

    from tvb_multiscale.tvb_nest.nest_models.devices import \
        NESTInputDeviceDict, NESTParrotSpikeInputDeviceDict, NESTOutputDeviceDict

    params = deepcopy(params)

    if nest_instance is None:
        nest_instance = load_nest(config=config)
        return_nest = True
    else:
        return_nest = False
    # Assert the model name...
    nest_device_model = device_to_dev_model(device_model)
    label = kwargs.pop("label", "")
    parrot = None
    input_device = False
    number_of_devices = int(params.pop("number_of_devices", 1))
    if device_model in NESTInputDeviceDict.keys() or nest_device_model in NESTInputDeviceDict.keys():
        input_device = True
        devices_dict = NESTInputDeviceDict
        if device_model in NESTParrotSpikeInputDeviceDict.keys() or device_model == "spike_generator":
            number_of_devices = int(params.pop("number_of_neurons", number_of_devices))
            if device_model in NESTParrotSpikeInputDeviceDict.keys():
                record_parrot = params.pop("record", None)
                parrot = nest_instance.Create("parrot_neuron", number_of_devices)
                parrot_connect_method = "one_to_one" if device_model == "parrot_spike_generator" else "all_to_all"
        default_params = config.NEST_INPUT_DEVICES_PARAMS_DEF.get(device_model,
                                                                  config.NEST_INPUT_DEVICES_PARAMS_DEF.get(
                                                                      deepcopy(nest_device_model, dict())))
    elif device_model in NESTOutputDeviceDict.keys():
        devices_dict = NESTOutputDeviceDict
        default_params = deepcopy(config.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(device_model,
                                                                   config.NEST_OUTPUT_DEVICES_PARAMS_DEF.get(
                                                                       deepcopy(nest_device_model, dict()))))
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.NEST_INPUT_DEVICES_PARAMS_DEF),
                           str(config.NEST_OUTPUT_DEVICES_PARAMS_DEF)))
    default_params["label"] = label
    if len(params) > 0:
        default_params.update(params)
    if input_device:
        label = default_params.pop("label", label)
    else:
        label = default_params.get("label", label)
        reset_upon_record = default_params.pop("reset_upon_record", False)
    # TODO: a better solution for the strange error with inhomogeneous poisson generator
    try:
        nest_device_node_collection = nest_instance.Create(nest_device_model, number_of_devices, params=default_params)
    except:
        warning("Using temporary hack for creating successive %s devices!" % device_model)
        nest_device_node_collection = nest_instance.Create(nest_device_model, number_of_devices, params=default_params)
    default_params["label"] = label
    if parrot:
        nest_device = devices_dict[device_model](nest_device_node_collection, parrot, nest_instance, label=label)
        # Connect the input spike device to the parrot neurons' population:
        nest_instance.Connect(nest_device.device, nest_device._nodes,
                              syn_spec={"weight": 1.0,
                                        "delay": nest_instance.GetKernelStatus("resolution"),
                                        "receptor_type": 0},
                              conn_spec={"rule": parrot_connect_method})
        if record_parrot is not None:
            rec_params = config.NEST_OUTPUT_DEVICES_PARAMS_DEF.get("spike_recorder", {})
            if isinstance(record_parrot, dict):
                record_parrot = deepcopy(record_parrot)
                rec_params.update(record_parrot)
            nest_device._record = nest_instance.Create("spike_recorder", params=rec_params)
            nest_instance.Connect(nest_device._nodes, nest_device._record)
    else:
        if input_device:
            nest_device = devices_dict[device_model](nest_device_node_collection, nest_instance, label=label)
        else:
            nest_device = devices_dict[device_model](nest_device_node_collection, nest_instance,
                                                     label=label, reset_upon_record=reset_upon_record)
    if return_nest:
        return nest_device, nest_instance
    else:
        return nest_device


def connect_device(nest_device, population, neurons_inds_fun, weight=1.0, delay=0.0, receptor_type=0,
                   syn_spec=None, conn_spec=None, config=CONFIGURED, **kwargs):
    """This method connects a NESTDevice to a NESTPopulation instance.
       Arguments:
        nest_device: the NESTDevice instance
        population: the NESTPopulation instance
        neurons_inds_fun: a function to return a NESTPopulation or a subset thereof of the target population.
                          Default = None.
        weight: the weights of the connection. Default = 1.0.
        delay: the delays of the connection. Default = 0.0.
        receptor_type: type of the synaptic receptor. Default = 0.
        config: configuration class instance. Default: imported default CONFIGURED object.
       Returns:
        the connected NESTDevice
    """
    from tvb_multiscale.tvb_nest.nest_models.devices import NESTParrotSpikeInputDeviceDict

    nest_instance = nest_device.nest_instance
    if receptor_type is None:
        receptor_type = 0
    if nest_instance is None:
        raise_value_error("There is no NEST instance!")
    resolution = nest_instance.GetKernelStatus("resolution")
    try:
        if delay < resolution:
            warning("Delay %f is smaller than the NEST simulation resolution %f!\n"
                    "Setting minimum delay equal to resolution!" % (delay, resolution))
            delay = resolution
    except:
        pass
    basic_syn_spec = {"weight": weight, "delay": delay, "receptor_type": receptor_type}
    if isinstance(syn_spec, dict):
        syn_spec = deepcopy(syn_spec)
        syn_spec.update(basic_syn_spec)
    else:
        syn_spec = basic_syn_spec
    neurons = get_populations_neurons(population, neurons_inds_fun)
    if nest_device.model == "spike_recorder":
        #                     source  ->  target
        nest_instance.Connect(neurons, nest_device.device, syn_spec=syn_spec)
    else:
        if isinstance(nest_device, tuple(NESTParrotSpikeInputDeviceDict.values())):
            # This is the case where we connect to the target neurons
            # the parrot_neuron population that is attached to the input spike device
            try:
                # TODO: Find a way to deal with this when delays are given as distributions
                # Try to reduce delay by resolution time
                syn_spec["delay"] = np.maximum(resolution, syn_spec["delay"] - resolution)
            except:
                pass
            if conn_spec is None:
                conn_spec = {"rule": "all_to_all"}
            else:
                conn_spec = deepcopy(conn_spec)
            conn_spec = create_conn_spec(n_src=nest_device.number_of_neurons, n_trg=len(neurons),
                                         src_is_trg=False, config=config, **conn_spec)[0]
            conn_spec.pop("allow_autapses", None)
            conn_spec.pop("allow_multapses", None)
            receptors = ensure_list(syn_spec["receptor_type"])
            for receptor in receptors:
                syn_spec["receptor_type"] = receptor
                nest_instance.Connect(nest_device._nodes, neurons, syn_spec=syn_spec, conn_spec=conn_spec)
        else:
            nest_instance.Connect(nest_device.device, neurons, syn_spec=syn_spec)
    return nest_device

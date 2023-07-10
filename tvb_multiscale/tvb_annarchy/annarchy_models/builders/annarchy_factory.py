# -*- coding: utf-8 -*-

import os
import importlib
from six import string_types

import numpy as np

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error, warning
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs, delete_folder_safely

from tvb_multiscale.core.spiking_models.devices import DeviceSets
from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDeviceDict, ANNarchyOutputDeviceDict, ANNarchyInputDevice, ANNarchyContinuousInputDevice


LOG = initialize_logger(__name__)


#TODO: Find a better way to abstract between nest_factory and factory!


# Helper functions with NEST

def load_annarchy(config=CONFIGURED, logger=LOG, clean_compilation_directory=True, **kwargs):
    """This function will load an ANNarchy instance and return it.
        Arguments:
         config: configuration class instance. Default: imported default CONFIGURED object.
         logger: logger object. Default: local LOG object.
         clean_compilation_directory: if True (default) the ANNarchy compilation directory
                                      will be deleted, if it exists, before importing ANNarchy
         **kwargs: other possible keyword arguments to run ANNarchy.setup(**kwargs)
        Returns:
         the imported ANNarchy instance
    """
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


def assert_model(model, annarchy_instance=None, import_path=""):
    """This function will confirm an ANNarchy model:
       Arguments:
        model: either a model name (string) or an ANNarchy.Neuron or ANNarchy Synapse model, or, even,
               a SpecificPopulation or SpecificProjection name
        annarchy_instance: the ANNarchy instance. Default = None.
        import_path: the path to be possibly searched to import the model. Default = ""
       Returns:
        ANNarchy model or the name of the SpecificPopulation or SpecificProjection
       One of the annarchy_instance or import_path must be given.
       We try to see if the model exists already in ANNarchy and return it (getattr(annarchy_instance, model_name),
       or, if the above fails, we try to import it from the import_path, assuming it already exists there
       else, we assume that it is already a ANNarchy Neuron or Synapse model and we return it.
       If the model corresponds to a SpecificPopulation or SpecificProjection, we retun only its name.
    """
    if isinstance(model, string_types):
        try:
            module = str(annarchy_instance.__dict__[model])
            if "Population" in module or "Projection" in module:
                # If this model is a SpecificPopulation or SpecificProjection, return its name:
                return model
            # ...try to load it from ANNarchy if it ships with its main distribution:
            model = getattr(annarchy_instance, model)
        except:
            # ...or otherwise import it from the users models' directory:
            model = importlib.import_module(model, import_path)
    # Return the model class
    return model


def set_model_parameters(model_instance, **params):
    for key, val in params.items():
        setattr(model_instance, key, val)
    return model_instance


def create_population(model, annarchy_instance, size=1, params={}, import_path="", config=CONFIGURED):
    """This function creates an ANNarchy.Population or ANNarchy Specific Population.
       Arguments:
        model: the model name (string)
        annarchy_instance: an ANNarchy module instance
        import_path: the path to be possibly searched to import the model. Default = ""
        size: The size of the population. Default = 1. It will not be used if geometry is given in the params.
        params: a dictionary of properties to be set for the ANNarchy.Population. Default = {}
       Returns:
        the ANNarchy.Population created and parametrized.
    """
    # Get either the Neuron class or the SpecificPopulation model name
    model = assert_model(model, annarchy_instance, import_path)
    if isinstance(model, string_types):
        model = getattr(annarchy_instance, model)

    # If model is a SpecificPopulation, create it directly:
    if model in [annarchy_instance.SpikeSourceArray,
                 annarchy_instance.TimedArray, annarchy_instance.TimedPoissonPopulation]:
        geometry = params.pop("geometry", 1)  # remove geometry argument for SpikeSourceArray and TimedArray
        if model == annarchy_instance.SpikeSourceArray:
            val = \
                params.pop("spike_times", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF["SpikeSourceArray"]["spike_times"])
            if len(val) == 1:
                val *= geometry
            population = model(val, **params)
        else:
            val = params.pop("rates", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF["TimedArray"]["rates"])
            if model == annarchy_instance.TimedArray:
                if val.shape[1] == 1:
                    val = np.repeat(val, geometry, axis=1)
                population = model(val, **params)
            else:  # TimedPoissonPopulation
                population = model(geometry, val, **params)
    elif model in [annarchy_instance.PoissonPopulation, annarchy_instance.HomogeneousCorrelatedSpikeTrains]:
        geometry = params.pop("geometry", size)
        if model == annarchy_instance.HomogeneousCorrelatedSpikeTrains:
            rates = params.pop("rates",
                               config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF["HomogeneousCorrelatedSpikeTrains"]["rates"])
            corr = params.pop("corr",
                              config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF["HomogeneousCorrelatedSpikeTrains"]["corr"])
            tau = params.pop("tau",
                             config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF["HomogeneousCorrelatedSpikeTrains"]["tau"])
            population = annarchy_instance.HomogeneousCorrelatedSpikeTrains(geometry, rates, corr, tau, **params)
        else:
            rates = params.pop("rates", None)
            if rates is None:
                target = params.pop("target", "exc")
            else:
                target = None
            population = annarchy_instance.PoissonPopulation(geometry, rates=rates, target=target, **params)
    else:
        population = annarchy_instance.Population(geometry=params.pop("geometry", size), neuron=model)
        # Parametrize the population:
        if len(params):
            population = set_model_parameters(population, **params)
    return population


def get_population_ind(annarchy_population, annarchy_instance):
    return annarchy_instance.Global._network[0]["populations"].index(annarchy_population)


def get_populations_neurons(population, inds_fun=None):
    """This function will return a ANNarchy Population or PopulationView instance
       of the ANNarchyPopulation._population, if inds_fun argument is a function
       Arguments:
        population: an ANNarchyPopulation class instance
        inds_fun: a function that takes an ANNarchy Population as argument and returns an ANNarchy PopulationView
       Returns:
        ANNarchy Population ANNarchyPopulation._population instance or an ANNarchy PopulationView thereof
    """
    if inds_fun is None:
        return population._nodes
    return inds_fun(population._nodes)


def get_proxy_target_pop(target_pop, input_device, neurons_inds_fun=None, import_path="", **kwargs):
    annarchy_instance = input_device.annarchy_instance
    population_to_connect_to = kwargs.get("proxy", None)  # in case proxy is provided already
    proxy_devices = kwargs.get("input_proxies", DeviceSets())
    if population_to_connect_to is None:
        # Check if proxy has been already created for this ANNArchyNetwork:
        proxy_label = "%s_proxy" % target_pop.label
        reg_label = target_pop.brain_region
        if proxy_label in proxy_devices.keys():
            if reg_label in proxy_devices[proxy_label]:
                population_to_connect_to = proxy_devices[proxy_label][reg_label]
        else:
            from tvb_multiscale.core.spiking_models.devices import DeviceSet
            proxy_devices[proxy_label] = DeviceSet(label=proxy_label, model=input_device.proxy_type)
    if population_to_connect_to is None:
        # Create the population and add it to this ANNArchyNetwork:
        population_to_connect_to = create_population(input_device.proxy_type, annarchy_instance,
                                                     size=target_pop.number_of_neurons,
                                                     params=kwargs.get("proxy_params", {}),
                                                     import_path=import_path)
        proxy_devices[proxy_label][reg_label] = population_to_connect_to
        # Connect the input proxy to the target population:
        if neurons_inds_fun is None:
            target_neurons = target_pop._nodes
        else:
            target_neurons = neurons_inds_fun(target_pop._nodes)
        proj = annarchy_instance.CurrentInjection(pre=population_to_connect_to,
                                                  post=target_neurons,
                                                  target=input_device.proxy_target)
        proj.connect_current()
    kwargs["input_proxies"] = proxy_devices
    if neurons_inds_fun is None:
        return population_to_connect_to
    else:
        return neurons_inds_fun(population_to_connect_to)


def connect_two_populations(source_pop, target_pop, weights=1.0, delays=0.0, target="exc", syn_spec={}, conn_spec={},
                            source_view_fun=None, target_view_fun=None, name=None, import_path="", **kwargs):
    """
    function to set up and connect a projection between two ANNarchyPopulations.
    Arguments:
      - souce_pop: The ANNarchyPopulation we want to connect from.
      - target_pop: The ANNarchyPopulation we want to connect to.
      - weights: the weights of the connection. Default = 1.0.
      - delays: the delays of the connection. Default = 0.0.
      - target: type of the synapse.. Default = "exc".
      - source_view_fun: a function to return an ANNarchy PopulationView of the source population.
                         Default = None.
      - target_view_fun: a function to return an ANNarchy PopulationView of the target population
                         Default = None.
      - syn_spec: optional dictionary of synaptic parameters. Default = {}
      - conn_spec: optional dictionary of connection parameters. Default = {}
      - name: name of the projection
      - import_path: the path to be possibly searched to import a model. Default = ""
      - **kwargs
      Returns: the projection
      """
    annarchy_instance = source_pop.annarchy_instance
    # Create the projection first
    source_neurons = get_populations_neurons(source_pop, source_view_fun)
    if isinstance(source_pop, ANNarchyContinuousInputDevice) and getattr(source_pop, "proxy", False):
        target_neurons = get_proxy_target_pop(target_pop, source_pop, target_view_fun, import_path, **kwargs)
    else:
        target_neurons = get_populations_neurons(target_pop, target_view_fun)
    if isinstance(syn_spec, dict):
        synapse = syn_spec.pop("synapse_model", syn_spec.pop("model", syn_spec.pop("synapse", None)))
    else:
        synapse = None
        syn_spec = {}
    if name is None:
        name = "%s -> %s" % (source_pop.label, target_pop.label)
    if isinstance(synapse, string_types):
        # If this is a SpecificProjection, create it directly:
        proj = getattr(annarchy_instance, synapse)(source_neurons, target_neurons, target=target, name=name, **syn_spec)
    else:
        # Otherwise, create it via the Projection creator:
        proj = set_model_parameters(annarchy_instance.Projection(source_neurons, target_neurons,
                                                                 target=target, synapse=synapse, name=name), **syn_spec)
    # Build the connection:
    if isinstance(conn_spec, dict):
        rule = conn_spec.pop("rule", "all_to_all").lower()
    else:
        rule = "all_to_all"
        conn_spec = {}
    if rule == "current":
        warning("Ignoring weight and delay for connect_current rule, for the connection %s -> %s!"
                % (source_pop.label, target_pop.label))
        proj = proj.connect_current()
    elif rule == "fixed_probability":
        proj = proj.connect_fixed_probability(conn_spec.pop("probability"), weights, delays=delays, **conn_spec)
    elif rule in ["fixed_number_pre", "fixed_number_post"]:
        proj = getattr(proj, "connect_" + rule)(conn_spec.pop("number"), weights, delays=delays, **conn_spec)
    elif rule == "gaussian":
        proj = proj.connect_gaussian(conn_spec.pop("amp"), conn_spec.pop("sigma"), delays=delays, **conn_spec)
    elif rule == "dog":
        proj = proj.connect_dog(conn_spec.pop("amp_pos"), conn_spec.pop("sigma_pos"),
                                conn_spec.pop("amp_neg"), conn_spec.pop("sigma_neg"),
                                delays=delays, **conn_spec)
    elif rule == "from_file":
        proj = proj.connect_from_file(conn_spec["filename"])
    elif rule == "with_func":
        proj = proj.connect_with_func(conn_spec.pop("rule"), **conn_spec)
    else:
        proj = getattr(proj, "connect_" + rule)(weights, delays=delays, **conn_spec)
    return proj


def params_dict_to_parameters_string(params):
    """This function will convert a dictionary of parameters to a parameters string argument
       to Neuron or Synapse creators of ANNarchy.
       Argument:
        params: a dictionary of parameter names and values
       Returns:
        parameters: string
    """
    parameters = ""
    for p, p_val in params.items():
        parameters += """%s=%f\n""" % (p, p_val)
    return parameters


def create_input_device(annarchy_device, import_path, params={}, config=CONFIGURED):
    """This functions populates an ANNarchyInputDevice instance with its device ANNarchy Population instance.
       Arguments:
        annarchy_device: a ANNarchyInputDevice instance
        params: a dictionary of devices' parameters. Default = {}
        config: configuration class instance. Default: imported default CONFIGURED object.
       Returns:
        annarchy_device: the same ANNarchyInputDevice instance populated
    """
    # if isinstance(annarchy_device, ANNarchyACCurrentInjector):
    #     # For the special case of ANNarchyACCurrentInjector convert a possible frequency parameter f,
    #     # to an angular frequency parameters omega = 2*pi*f
    #     f = params.pop("frequency", params.pop("freq", params.pop("f", None)))
    #     if f:
    #         params["omega"] = 2 * np.pi * f
    number_of_neurons = params.pop("number_of_neurons", None)
    if number_of_neurons is not None:
        params["geometry"] = number_of_neurons
    record = params.pop("record", None)
    annarchy_device._nodes = create_population(annarchy_device.model, annarchy_device.annarchy_instance,
                                               params=params, import_path=import_path, config=config)
    annarchy_device.device = annarchy_device._nodes
    annarchy_device._nodes.name = annarchy_device.label
    if record is not None:
        rec_params = {}
        if isinstance(record, dict):
            rec_params = list(record.values())[0]
            record = list(record.keys())[0]
        annarchy_device._record = \
            annarchy_device.annarchy_instance.Monitor(annarchy_device._nodes, record, **rec_params)
    return annarchy_device


def create_device(device_model, params=None, config=CONFIGURED, annarchy_instance=None, **kwargs):
    """function to create an ANNarchyInputDevice or ANNarchyOutputDevice.
       The device will be only created for ANNarchyOutputDevice and also populated for ANNarchyInputDevice.
       Arguments:
        device_model: name (string) of the device model
        params: dictionary of parameters of device and/or its synapse. Default = None
        config: configuration class instance. Default: imported default CONFIGURED object.
        annarchy_instance: the ANNarchy instance.
                           Default = None, in which case we are going to load one, and also return it in the output
         - **kwargs
       Returns:
        the ANNarchyDevice class, and optionally, the ANNarchy instance if it is loaded here.
    """
    if annarchy_instance is None:
        annarchy_instance = load_annarchy(config=config)
        return_annarchy = True
    else:
        return_annarchy = False
    # Figure out if this is an input or an output device:
    label = kwargs.pop("label", "")
    # Get the default parameters for this device...
    if device_model in ANNarchyInputDeviceDict.keys():
        devices_dict = ANNarchyInputDeviceDict
        default_params = config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF.get(device_model, {}).copy()
        if len(label):
            default_params["name"] = label
    elif device_model in ANNarchyOutputDeviceDict.keys():
        devices_dict = ANNarchyOutputDeviceDict
        default_params = config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF.get(device_model, {}).copy()
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF),
                           str(config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF)))
    # ...and update them with any user provided parameters
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
        label = default_params.get("name", default_params.pop("label", label))
    # Create the ANNarchy Device class:
    annarchy_device = devices_dict[device_model](None, annarchy_instance=annarchy_instance, label=label)
    if isinstance(annarchy_device, ANNarchyInputDevice):
        if isinstance(annarchy_device, ANNarchyContinuousInputDevice):
            annarchy_device.proxy = default_params.pop("proxy", annarchy_device.proxy)
            annarchy_device.proxy_type = default_params.pop("proxy_type", annarchy_device.proxy_type)
            annarchy_device.proxy_target = default_params.pop("proxy_target", annarchy_device.proxy_target)
        # If it is an input device, populate it:
        annarchy_device = create_input_device(annarchy_device,
                                              kwargs.get("import_path", config.MYMODELS_IMPORT_PATH),
                                              default_params.copy(), config)
    annarchy_device.params = default_params.copy()
    if return_annarchy:
        return annarchy_device, annarchy_instance
    else:
        return annarchy_device


def connect_input_device(annarchy_device, population, neurons_inds_fun=None,
                         weight=1.0, delay=0.0, receptor_type="exc", syn_spec=None, conn_spec=None,
                         import_path=CONFIGURED.MYMODELS_IMPORT_PATH, **kwargs):
    """This function connect an ANNarchyInputDevice to an ANNarchyPopulation instance.
       Arguments:
        annarchy_device: the ANNarchyInputDevice instance
        population: the ANNarchyPopulation instance
        neurons_inds_fun: a function to return an ANNarchy PopulationView of the target population. Default = None.
        weight: the weights of the connection. Default = 1.0
        delay: the delays of the connection. Default = 0.0
        receptor_type: type of the synapse (target in ANNarchy). Default = "exc"
        import_path: the path to be possibly searched to import the model. Default is taken from CONFIGURED
       Returns:
        the connected ANNarchyInputDevice
    """
    if isinstance(conn_spec, dict):
        connection_args = conn_spec
    else:
        connection_args = {}
    if isinstance(syn_spec, dict):
        synapse = syn_spec.pop("synapse_model",
                               syn_spec.pop("model",
                                          syn_spec.pop("synapse", None)))
    else:
        syn_spec = {}
        synapse = None
    if synapse is not None:
        syn_spec["synapse"] = assert_model(synapse, annarchy_device.annarchy_instance, import_path)
    proj = connect_two_populations(annarchy_device, population,
                                   weight, delay, receptor_type, syn_spec, connection_args,
                                   source_view_fun=None, target_view_fun=neurons_inds_fun,
                                   import_path=import_path, **kwargs)
    # Add this projection to the source device's and target population's inventories:
    annarchy_device.projections_pre.append(proj)
    population.projections_post.append(proj)
    return annarchy_device


def connect_output_device(annarchy_device, population, neurons_inds_fun=None):
    """This function connect an ANNarchyOutputDevice to an ANNarchyPopulation instance,
       by also populating it with an ANNarchy Monitor.
           Arguments:
            annarchy_device: the ANNarchyInputDevice instance
            population: the ANNarchyPopulation instance
            neurons_inds_fun: a function to return an ANNarchy PopulationView of the target population. Default = None.
           Returns:
            the connected ANNarchyOutputDevice
        """
    neurons = get_populations_neurons(population, neurons_inds_fun)
    params = annarchy_device.params.copy()
    # Create a connection by adding an ANNarchy Monitor targeting the specific neurons of this population:
    monitor = annarchy_device.annarchy_instance.Monitor(neurons, **params)
    monitor.name = "%s_%d" % (annarchy_device.label, len(annarchy_device.monitors) + 1)
    annarchy_device.monitors[monitor] = neurons
    annarchy_device.device = annarchy_device.monitors
    return annarchy_device


def connect_device(annarchy_device, population, neurons_inds_fun=None,
                   weight=1.0, delay=0.0, receptor_type="exc", syn_spec=None, conn_spec=None,
                   config=CONFIGURED, **kwargs):
    """This function connects an ANNarchyInputDevice or an ANNarchyOutputDevice to an ANNarchyPopulation instance.
       If the device is an ANNarchyOutputDevice, it will also be populated by an ANNarchy Monitor.
        The arguments weight, delay and receptor_type are ignored for output devices
        (i.e., that are based on ANNarchy Monitors)
           Arguments:
            annarchy_device: the ANNarchy Device instance
            population: the ANNarchyPopulation instance
            neurons_inds_fun: a function to return an ANNarchy PopulationView of the target population. Default = None.
            weight: the weights of the connection. Default = 1.0
            delay: the delays of the connection. Default = 0.0
            receptor_type: type of the synapse (target in ANNarchy). Default = "exc".
            config: configuration class instance. Default: imported default CONFIGURED object.
           Returns:
            the connected ANNarchy Device
        """
    if isinstance(annarchy_device, ANNarchyInputDevice):
        return connect_input_device(annarchy_device, population, neurons_inds_fun, weight, delay, receptor_type,
                                    syn_spec=syn_spec, conn_spec=conn_spec,
                                    import_path=kwargs.pop("import_path", config.MYMODELS_IMPORT_PATH),
                                    **kwargs)
    else:
        return connect_output_device(annarchy_device, population, neurons_inds_fun)

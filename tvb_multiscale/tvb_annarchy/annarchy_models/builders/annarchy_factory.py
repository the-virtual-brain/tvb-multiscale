# -*- coding: utf-8 -*-

import os
import importlib
from six import string_types
import warnings

import numpy as np

import ANNarchy

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error, warning
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list
from tvb.contrib.scripts.utils.file_utils import safe_makedirs, delete_folder_safely

from tvb_multiscale.core.utils.data_structures_utils import \
    safe_deepcopy, repeat_nested_lists_to_size, repeat_to_shape_robust
from tvb_multiscale.core.spiking_models.devices import DeviceSets
from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.population import ANNarchyPopulation
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDeviceDict, ANNarchyOutputDeviceDict, ANNarchyInputDevice, ANNarchyTimedArrayToSpikes


LOG = initialize_logger(__name__)


# Helper functions with NEST

def load_annarchy(logger=LOG, clean_compilation_directory=True):
    """This function will load an ANNarchy instance and return it.
        Arguments:
         logger: logger object. Default: local LOG object.
         clean_compilation_directory: if True (default) the ANNarchy compilation directory
                                      will be deleted, if it exists, before importing ANNarchy
        Returns:
         the imported ANNarchy instance
    """
    logger.info("Loading an ANNarchy instance...")
    if clean_compilation_directory:
        logger.info("Cleaning ANNarchy compilation directory, if any...")
        delete_folder_safely(os.path.join(os.getcwd(), "annarchy"))
    ANNarchy.clear()
    return ANNarchy


def assert_model(model, import_path=""):
    """This function will confirm an ANNarchy model:
       Arguments:
        model: either a model name (string) or an ANNarchy.Neuron or ANNarchy Synapse model, or, even,
               a SpecificPopulation or SpecificProjection name
        import_path: the path to be possibly searched to import the model. Default = ""
       Returns:
        ANNarchy model or the name of the SpecificPopulation or SpecificProjection
       We try to see if the model exists already in ANNarchy and return it (getattr(ANNarchy, model_name),
       or, if the above fails, we try to import it from the import_path, assuming it already exists there
       else, we assume that it is already a ANNarchy Neuron or Synapse model and we return it.
       If the model corresponds to a SpecificPopulation or SpecificProjection, we retun only its name.
    """
    if isinstance(model, string_types):
        try:
            module = str(ANNarchy.__dict__[model])
            if "Population" in module or "Projection" in module:
                # If this model is a SpecificPopulation or SpecificProjection, return its name:
                return model
            # ...try to load it from ANNarchy if it ships with its main distribution:
            model = getattr(ANNarchy, model)
        except:
            # ...or otherwise import it from the users models' directory:
            model = importlib.import_module(model, import_path)
    # Return the model class
    return model


def set_model_parameters(model_instance, **params):
    for key, val in params.items():
        setattr(model_instance, key, val)
    return model_instance


def create_population(model, annarchy_network, size=1, params=dict(), import_path="", config=CONFIGURED):
    """This function creates an ANNarchy Population or ANNarchy Specific Population.
       Arguments:
        model: the model name (string)
        annarchy_network: an ANNarchy Network instance
        import_path: the path to be possibly searched to import the model. Default = ""
        size: The size of the population. Default = 1. It will not be used if geometry is given in the params.
        params: a dictionary of properties to be set for the ANNarchy.Population. Default = {}
       Returns:
        the ANNarchy.core.Population.Population created and parametrized.
    """
    # Get either the Neuron class or the SpecificPopulation model name
    params = safe_deepcopy(params)
    model = assert_model(model, import_path)
    if isinstance(model, string_types):
        model_name = model
        model = getattr(ANNarchy, model)
    else:
        try:
            # For neurons:
            model_name = model.name
        except:
            # For input populations:
            model_name = model.__name__
    # If model is a SpecificPopulation, create it directly:
    if model in [ANNarchy.inputs.SpikeSourceArray,
                 ANNarchy.PoissonPopulation,
                 ANNarchy.inputs.TimedArray,
                 ANNarchy.inputs.TimedPoissonPopulation,
                 ANNarchy.inputs.HomogeneousCorrelatedSpikeTrains]:
        geometry = params.pop("geometry", None)  # remove geometry argument for SpikeSourceArray and TimedArray
        if model == ANNarchy.inputs.SpikeSourceArray:
            rates = params.pop("spike_times",
                             config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF[model_name]["spike_times"])
            if geometry is not None:
                rates = repeat_nested_lists_to_size(rates, np.prod(geometry))
            population = annarchy_network.create(model(rates, **params))
        elif model == ANNarchy.PoissonPopulation:
            rates = params.pop("rates", None)
            if rates is None:
                rates = params.pop("rates", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF[model_name]["rates"])
                target = params.pop("target", "exc")
            else:
                target = None
            if geometry is None:
                geometry = 1
            population = annarchy_network.create(
                ANNarchy.PoissonPopulation(geometry, rates=rates, target=target, **params))
        else:
            schedule = params.pop("schedule", None)
            n_step = params.pop("n_step", None)
            rates = params.pop("rates", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF[model_name]["rates"])
            if rates.ndim < 1:
                rates = np.atleast_1d(rates)
            elif rates.ndim < 2:
                rates = rates[:, np.newaxis]
            if geometry is None:
                geometry_shape = rates.shape[1:]
                geometry = geometry_shape
            else:
                geometry_shape = tuple(ensure_list(geometry))
            if schedule is None:
                if n_step is None:
                    n_step = rates.shape[0]
                schedule_shape = (n_step,)
                schedule = np.arange(n_step).tolist()
            else:
                schedule = ensure_list(schedule)
                n_schedule = len(schedule)
                if n_step is not None:
                    if n_step > 1 and n_schedule == 1:
                        warnings.warn("User argument n_step = %d is used to overwrite user argument schedule = %s, "
                                      "of length %d!" % (n_step, str(schedule), n_schedule))
                        n_schedule = n_step
                        schedule = np.arange(n_step).tolist()
                    elif n_step != n_schedule:
                        raise ValueError("The length of user argument schedule (%d) is "
                                         "neither 1 nor equal to user argument n_step (%d)" % (n_schedule, n_step))
                else:
                    n_step = n_schedule
                schedule_shape = (n_step,)
            if model == ANNarchy.inputs.HomogeneousCorrelatedSpikeTrains:
                rates = ensure_list(rates.flatten())
                if len(rates) == 1:
                    rates *= n_step
                elif rates.size != n_step:
                    raise ValueError("The length of user argument rates (%d) is "
                                     "neither 1 nor equal to the length of schedule (%d)" % (len(rates), n_step))
            else:
                rates = repeat_to_shape_robust(rates, schedule_shape + geometry_shape) # target shape
            if model == ANNarchy.inputs.HomogeneousCorrelatedSpikeTrains:
                params["corr"] = params.get("corr", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF[model_name]["corr"])
                params["tau"] = params.get("tau", config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF[model_name]["tau"])
            population = annarchy_network.create(model(rates=rates, geometry=geometry, schedule=schedule, **params))
    else:
        population = annarchy_network.create(params.pop("geometry", size), neuron=model)
        # Parametrize the population:
        if len(params):
            population = set_model_parameters(population, **params)
    return population


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


def connect_two_populations(source_pop, target_pop, weights=1.0, delays=0.0, target="exc", syn_spec={}, conn_spec={},
                            source_view_fun=None, target_view_fun=None, name=None):
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
      Returns: the projection
      """
    annarchy_network = source_pop.annarchy_network
    # Create the projection first
    source_neurons = get_populations_neurons(source_pop, source_view_fun)
    target_neurons = get_populations_neurons(target_pop, target_view_fun)
    if isinstance(syn_spec, dict):
        syn_spec = safe_deepcopy(syn_spec)
        synapse = syn_spec.pop("synapse_model", syn_spec.pop("model", syn_spec.pop("synapse", None)))
    else:
        synapse = None
        syn_spec = dict()
    if name is None:
        name = "%s -> %s" % (source_pop.label, target_pop.label)
    if isinstance(synapse, string_types):
        # If this is a SpecificProjection, create it directly:
        proj = annarchy_network.connect(
            getattr(ANNarchy, synapse)(source_neurons, target_neurons, target=target, name=name, **syn_spec))
    elif synapse is ANNarchy.inputs.CurrentInjection:
        proj = annarchy_network.connect(
            synapse(source_neurons, target_neurons, target=target, name=name, **syn_spec))
    else:
        # Otherwise, create it via the Projection creator:
        proj = set_model_parameters(
                    annarchy_network.connect(source_neurons, target_neurons,
                                             target=target, synapse=synapse, name=name), **syn_spec)
    # Build the connection:
    if isinstance(conn_spec, dict):
        conn_spec = safe_deepcopy(conn_spec)
        rule = conn_spec.pop("rule", "all_to_all").lower()
    else:
        rule = "all_to_all"
        conn_spec = dict()
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


def create_input_device(annarchy_device, annarchy_network, params=dict(), import_path="", config=CONFIGURED):
    """This functions populates an ANNarchyInputDevice instance with its device ANNarchy Population instance.
       Arguments:
        annarchy_device: a ANNarchyInputDevice instance
        annarchy_network: Instance of ANNarchy Network
        params: a dictionary of devices' parameters. Default = {}
        import_path: the path to be possibly searched to import a model. Default = ""
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
    params = safe_deepcopy(params)
    number_of_neurons = params.pop("number_of_neurons", None)
    if number_of_neurons is not None:
        params["geometry"] = number_of_neurons
    record = params.pop("record", None)
    proxy = params.pop("proxy", None)
    proxy_params = params.pop("proxy_params", dict())
    geometry = params.get("geometry", 1)
    annarchy_device._nodes = create_population(annarchy_device.model, annarchy_device.annarchy_network,
                                               size=geometry, params=params, import_path=import_path, config=config)
    annarchy_device.device = annarchy_device._nodes
    annarchy_device._nodes.name = annarchy_device.label
    if isinstance(record, dict):
        record = safe_deepcopy(record)
        record_from = record.pop("from")
        annarchy_device.record = \
            annarchy_device.annarchy_network.monitor(annarchy_device._nodes, record_from, **record)
    if isinstance(annarchy_device, ANNarchyTimedArrayToSpikes):
        annarchy_device.proxy_params = dict(safe_deepcopy(proxy_params))
        annarchy_device.proxy_target = annarchy_device.proxy_params.pop("target",
                                                                        annarchy_device.proxy_target)
        record = annarchy_device.proxy_params.pop("record", None)
        if proxy is None:
            proxy_fun = proxy_params.pop("proxy_fun", None)
            if proxy_fun is not None:
                if len(annarchy_device.brain_region):
                    label = "%s-%s" % (annarchy_device.label, annarchy_device.brain_region)
                else:
                    label = str(annarchy_device.label)
                proxy, populations, projections, monitors = proxy_fun(label, annarchy_network, **proxy_params)
                annarchy_device.proxy_populations = populations
                annarchy_device.proxy_projections = projections
                annarchy_device.proxy_monitors = monitors
            else:
                proxy = create_population(annarchy_device.proxy_params.pop("model"),
                                          annarchy_network, size=geometry,
                                          params=annarchy_device.proxy_params,
                                          import_path=annarchy_device.proxy_params.pop("import_path", ""),
                                          config=config)
        else:
            annarchy_device.proxy_populations = proxy_params.pop("proxy_populations", dict())
            annarchy_device.proxy_projections = proxy_params.pop("proxy_projections", dict())
            annarchy_device.proxy_monitors = proxy_params.pop("proxy_monitors", dict())
        if not isinstance(proxy, ANNarchyPopulation):
            proxy = ANNarchyPopulation(proxy, annarchy_network,
                                       model=proxy.neuron_type.name,
                                       label=annarchy_device.label,
                                       brain_region=annarchy_device.brain_region)
        annarchy_device.proxy = proxy
        annarchy_device.model = "TimedArrayTo%s" % annarchy_device.proxy.model
        proj = annarchy_network.connect(
                    ANNarchy.CurrentInjection(
                        annarchy_device.device, annarchy_device.proxy._nodes, annarchy_device.proxy_target))
        proj.connect_current()
        # Add this projection to the source device's and target population's inventories:
        annarchy_device.projections_pre.append(proj)
        annarchy_device.proxy.projections_post.append(proj)
        if isinstance(record, dict):
            record = safe_deepcopy(record)
            record_from = record.pop("from")
            annarchy_device.proxy.record = \
                annarchy_device.annarchy_network.monitor(annarchy_device.proxy._nodes, record_from, **record)
    return annarchy_device


def create_device(device_model, params=None, config=CONFIGURED, annarchy_network=None, **kwargs):
    """function to create an ANNarchyInputDevice or ANNarchyOutputDevice.
       The device will be only created for ANNarchyOutputDevice and also populated for ANNarchyInputDevice.
       Arguments:
        device_model: name (string) of the device model
        params: dictionary of parameters of device and/or its synapse. Default = None
        config: configuration class instance. Default: imported default CONFIGURED object.
        annarchy_network: An ANNarchy Network instance
         - **kwargs
       Returns:
        the ANNarchyDevice class, and optionally, the ANNarchy instance if it is loaded here.
    """
    if not isinstance(annarchy_network, ANNarchy.Network):
        raise ValueError("ANNarchy.Network argument is not provided!")
    # Figure out if this is an input or an output device:
    label = kwargs.pop("label", "")
    # Get the default parameters for this device...
    if device_model in ANNarchyInputDeviceDict:
        devices_dict = safe_deepcopy(ANNarchyInputDeviceDict)
        default_params = config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF.get(device_model, dict()).copy()
        if len(label):
            default_params["name"] = label
    elif device_model in ANNarchyOutputDeviceDict:
        devices_dict = safe_deepcopy(ANNarchyOutputDeviceDict)
        default_params = config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF.get(device_model, dict()).copy()
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(ANNarchyInputDeviceDict), str(ANNarchyOutputDeviceDict)))
    # ...and update them with any user provided parameters
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
        label = default_params.get("name", default_params.pop("label", label))
    # Create the ANNarchy Device class:
    annarchy_device = devices_dict[device_model](None, annarchy_network, label=label)
    if isinstance(annarchy_device, ANNarchyInputDevice):
        # If it is an input device, populate it:
        annarchy_device = create_input_device(annarchy_device, annarchy_network,
                                              safe_deepcopy(default_params),
                                              kwargs.pop("import_path", config.MYMODELS_IMPORT_PATH),
                                              config)
    annarchy_device.params = safe_deepcopy(default_params)
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
        syn_spec = safe_deepcopy(syn_spec)
        synapse = syn_spec.pop("synapse_model",
                               syn_spec.pop("model",
                                          syn_spec.pop("synapse", None)))
    else:
        syn_spec = dict()
        synapse = None
    if synapse is not None:
        syn_spec["synapse"] = assert_model(synapse, import_path)
    if isinstance(annarchy_device, ANNarchyTimedArrayToSpikes):
        proj = connect_two_populations(annarchy_device.proxy, population,
                                       weight, delay, receptor_type, syn_spec, connection_args,
                                       source_view_fun=None, target_view_fun=neurons_inds_fun)
        # Add this projection to the source device's proxy's inventory:
        annarchy_device.proxy.projections_pre.append(proj)
    else:
        proj = connect_two_populations(annarchy_device, population,
                                       weight, delay, receptor_type, syn_spec, connection_args,
                                       source_view_fun=None, target_view_fun=neurons_inds_fun)
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
    params = safe_deepcopy(annarchy_device.params)
    # Create a connection by adding an ANNarchy Monitor targeting the specific neurons of this population:
    monitor = annarchy_device.annarchy_network.monitor(neurons, **params)
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

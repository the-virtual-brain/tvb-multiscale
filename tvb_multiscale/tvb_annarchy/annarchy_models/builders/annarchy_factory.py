# -*- coding: utf-8 -*-

import os
import importlib
from six import string_types

import numpy as np

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDeviceDict, ANNarchyOutputDeviceDict, ANNarchyInputDevice, ANNarchyOutputDevice, \
    ANNarchySpikeSourceArray,ANNarchyPoissonPopulation, ANNarchyHomogeneousCorrelatedSpikeTrains, \
    ANNarchyCurrentInjector, ANNarchyDCCurrentInjector, ANNarchyACCurrentInjector

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


def assert_model(model, import_path, annarchy_instance):
    if isinstance(model, string_types):
        # If the model is actually a model name:
        model_name = model
        try:
            # ...try to load it from ANNarchy if it ships with its main distribution:
            model = getattr(annarchy_instance, model_name)
        except:
            # ...or otherwise import it from the users models' directory:
            model = importlib.import_module(model_name, import_path)
    else:
        model_name = model.name
    # Return the model class and name:
    return model, model_name


def get_populations_neurons(population, inds_fun=None):
    if inds_fun is None:
        return population._population
    return inds_fun(population._population)


def connect_two_populations(source_pop, target_pop, weights, delays, target,
                            source_view_fun=None, target_view_fun=None,
                            synapse=None, method="connect_all_to_all", name=None,
                            annarchy_instance=None, **connection_args):
    """
    Method to set up and connect a projection between two ANNarchyPopulations.
    Arguments:
      - souce_pop: The ANNarchyPopulation we want to connect from.
      - target_pop: The ANNarchyPopulation we want to connect to.
      - weights: the weights of the connection
      - delays: the delays of the connection
      - target: type of the connection. Needs to be set, or weights are zero.
      - source_view_fun: a function to return an ANNarchy PopulationView of the source population.
                         Default = None.
      - target_view_fun: a function to return an ANNarchy PopulationView of the target population
                         Default = None.
      - synapse: an ANarchy.Synapse instance.
      - name: name of the projection
      - method: name of an ANNarchy connection method
      - **connection_args: depend on the chosen ANNarchy connection method
      Returns: the projection
      """
    # Create the projection first
    proj = annarchy_instance.Projection(
                get_populations_neurons(source_pop._population, source_view_fun),
                get_populations_neurons(target_pop._population, target_view_fun),
                target=target, synapse=synapse, name=name)
    # Add this projection to the source and target population inventories:
    source_pop._projections_pre.append(proj)
    target_pop._projections_post.append(proj)
    # Build the connection:
    getattr(proj, method)(weights=weights, delays=delays, **connection_args)


def create_input_device(annarchy_device):
    if isinstance(annarchy_device, ANNarchyInputDevice):
        geometry = annarchy_device.params.get("geometry", 1)
        neuron_model = annarchy_device.params.get("neuron", None)
        parameters = annarchy_device.params.get("parameters", "")
        equations = annarchy_device.params.get("equations")
        label = annarchy_device.params.get("label", "")
        if isinstance(annarchy_device, ANNarchyCurrentInjector):
            connect_method = "connect_current"
            device_population = \
                create_current_injector(annarchy_device, geometry, neuron_model,
                                        equations, parameters, annarchy_device.annarchy_instance, label)
            create_current_injector(annarchy_device, geometry, neuron_model, parameters, annarchy_instance)
        else:
            connect_method = "connect_one_to_one"
            if isinstance(annarchy_device, ANNarchyPoissonPopulation):
                device_population = \
                    create_poisson_population(annarchy_device, geometry, neuron_model,
                                              equations, parameters, annarchy_device.annarchy_instance, label)
            elif isinstance(annarchy_device, ANNarchySpikeSourceArray):
                device_population = \
                    create_spike_source_array(annarchy_device, geometry, neuron_model,
                                              equations, parameters, annarchy_device.annarchy_instance, label)


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
        device_model = device_name
    if device_model in ANNarchyInputDeviceDict.keys():
        devices_dict = ANNarchyInputDeviceDict
        default_params_dict = config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF
    elif device_model in ANNarchyOutputDeviceDict.keys():
        devices_dict = ANNarchyOutputDeviceDict
        default_params_dict = config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF),
                           str(config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF)))
    default_params = dict(default_params_dict.get(device_name, {}))
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)
    label = default_params.pop("label", "")
    annarchy_device = devices_dict[device_name](None, label=label,
                                                params=default_params,
                                                anarchy_instance=annarchy_instance)
    if return_annarchy:
        return annarchy_device, annarchy_instance
    else:
        return annarchy_device

def params_dict_to_parameters_string(params):
    parameters = ""
    for p, p_val in params.items():
        parameters += "%s=%f\n" % (p, p_val)
    return parameters


def create_spike_source_array(annarchy_device, geometry, neuron_model,
                            equations, parameters, annarchy_instance, label=""):
    pass

def create_poisson_population(annarchy_device, geometry, neuron_model,
                              equations, parameters, annarchy_instance, label=""):
    pass



def create_current_injector(annarchy_device, geometry, neuron_model,
                            equations, parameters, annarchy_instance, label=""):
    if not isinstance(neuron_model, annarchy_instance.Neuron):
        if isinstance(parameters, dict):
            if isinstance(annarchy_device, ANNarchyACCurrentInjector):
                f = parameters.pop("f", None)
                if f:
                    parameters["omega"] = 2 * np.pi * f
            params = params_dict_to_parameters_string(parameters)
        else:
            params = parameters
        neuron_model = annarchy_instance.Neuron(equations=equations,
                                                parameters=params)
    return annarchy_instance.Population(geometry=geometry, neuron=neuron_model, name=label)


def connect_device(annarchy_device, neurons, neurons_inds_fun=None,
                   weight=1.0, delay=0.0, receptor_type="exc", config=CONFIGURED,
                   annarchy_instance=None):
    if annarchy_instance is None:
        raise_value_error("There is no NEST instance!")
    if neurons_inds_fun:
        neurons_selection = get_populations_neurons(neurons, neurons_inds_fun)
    else:
        neurons_selection = neurons
    resolution = annarchy_instance.dt()
    if isinstance(delay, annarchy_instance.Uniform):
        if delay.min < resolution:
            delay.min = resolution
            warning("Minimum delay %f is smaller than the NEST simulation resolution %f!\n"
                    "Setting minimum delay equal to resolution!" % (delay.min, resolution))
        if delay.max <= delay.min:
            raise_value_error("Maximum delay %f is not smaller than minimum one %f!" % (delay.max, delay.min))
    else:
        if delay < resolution:
            delay = resolution
            warning("Delay %f is smaller than the NEST simulation resolution %f!\n"
                    "Setting minimum delay equal to resolution!" % (delay, resolution))

            connect_two_populations(device_population, neurons_selection, weight, delay, receptor_type,
                                    method=connect_method, annarchy_instance=annarchy_instance)
    elif isinstance(annarchy_device, ANNarchyOutputDevice):
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (str(annarchy_device), str(config.ANNARCHY_INPUT_DEVICES_PARAMS_DEF),
                           str(config.ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF)))
    # nest_device.update_number_of_connections()
    return annarchy_device
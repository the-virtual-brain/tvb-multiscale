# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.population import SpikingPopulation

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, list_of_dicts_to_dicts_of_ndarrays


class ANNarchyPopulation(SpikingPopulation):

    annarchy_instance = None
    population_neurons = None # A population object
    projections_pre = []
    projections_post = []
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, population_neurons, label="", model="", annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        self.population_neurons = population_neurons
        super(ANNarchyPopulation, self).__init__(label, model, **kwargs)

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @property
    def neurons(self):  # tuple of populations' neurons
        """Method to get all neurons' indices of this population.
           Returns:
            tuple of neurons'indices.
            In ANNarchy: So far we get only local indices.
        """
        return tuple(self.population_neurons.ranks)

    def _assert_annarchy(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _print_neurons(self):
        """ Prints indices of neurons in this population.
            Currently we get only local indices.
        """
        return "\n%s" % str(self.population_neurons.ranks)

    def _Set(self, neurons, values_dict):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            neurons: tuple of neurons the attributes of which should be set.
            values_dict: dictionary of attributes names' and values.
        """
        self._assert_annarchy()
        self.population_neurons[neurons].set(values_dict)

    def _Get(self, neurons, attrs=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            neurons: tuple of neurons which should be included in the output.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of lists of neurons' attributes.
        """
        self._assert_annarchy()
        dictionary = {}
        if attrs is None:
            # If no attribute is specified, return all of them
            for attribute in self.population_neurons[neurons].attributes:
                dictionary[attribute] = self.population_neurons[neurons].get(attribute).tolist()
        else:
            for attribute in attrs:
                dictionary[attribute] = self.population_neurons[neurons].get(attribute).tolist()
        return dictionary

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
        Arguments:
            neurons: tuple of neurons the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            synapses' collections.
        """
        self._assert_annarchy()
        if neurons is not None:
            if len(neurons) == 0:
                neurons = None
        else:
            neurons = self.population_neurons.ranks
        if neurons is not None and not isinstance(neurons, self.annarchy_instance.NodeCollection):
            neurons = self.annarchy_instance.NodeCollection(neurons)
        if source_or_target not in ["source", "target"]:
            return self.annarchy_instance.GetConnections(source=neurons), \
                   self.annarchy_instance.GetConnections(target=neurons)
        else:
            kwargs = {source_or_target: neurons}
            return self.annarchy_instance.GetConnections(**kwargs)

    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             connections: connections' objects.
             values_dict: dictionary of attributes names' and values.
        """
        self._assert_annarchy()
        connections.set(values_dict)

    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            Returns:
             Dictionary of arrays of connections' attributes.

        """
        self._assert_annarchy()
        if attrs is None:
            return connections.get()
        else:
            return connections.get(attrs)

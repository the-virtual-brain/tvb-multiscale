# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.population import SpikingPopulation

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, list_of_dicts_to_dicts_of_ndarrays


class ANNarchyPopulation(SpikingPopulation):

    annarchy_instance = None
    projections_in = []  # incoming projections
    projections_out = []  # outgoing projections
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, population_neurons, label="", model="", annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        self._population = population_neurons
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
        return tuple(self._population.ranks)

    @property
    def population(self):
        return self._population

    def connect(self, target_pop, target, synapse=None, name=None,
                method="connect_all_to_all", **connection_args):
        """
        Method to set up and connect a projection between two ANNarchyPopulations.
        Arguments:
            target_pop: The ANNarchyPopulation we want to connect to.
            target: type of the connection. Needs to be set, or weights are zero.
            synapse: a ``Synapse`` instance.
            name: name of the projection
            method: name of an ANNarchy connection method
            **connection_args: depend on the chosen ANNarchy connection method
        Returns: the projection
        """
        proj = self.annarchy_instance.Projection(self._population, target_pop._population, target, synapse, name)
        self.projections_out.append(proj)  # is there a faster way than using .append()?
        target_pop.projections_in.append(proj)
        return getattr(proj, method)(**connection_args)

    def connect_from(self, source_pop, target, synapse=None, name=None,
                method="connect_all_to_all", **connection_args):
        """
        Method to set up and connect a projection between two ANNarchyPopulations.
        Arguments:
            source_pop: The ANNarchyPopulation we want to connect from.
            target: type of the connection. Needs to be set, or weights are zero.
            synapse: a ``Synapse`` instance.
            name: name of the projection
            method: name of an ANNarchy connection method
            **connection_args: depend on the chosen ANNarchy connection method
        Returns: the projection
        """
        proj = self.annarchy_instance.Projection(source_pop.population, self._population, target, synapse, name)
        self.projections_in.append(proj)  # is there a faster way than using .append()?
        source_pop.projections_out.append(proj)
        return getattr(proj, method)(**connection_args)

    def _assert_annarchy(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _print_neurons(self):
        """ Prints indices of neurons in this population.
            Currently we get only local indices.
        """
        return "\n%s" % str(self._population.ranks)

    def _Set(self, neurons, values_dict):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            neurons: tuple of neurons the attributes of which should be set.
            values_dict: dictionary of attributes names' and values.
        """
        self._assert_annarchy()
        self._population[neurons].set(values_dict)

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
            for attribute in self._population[neurons].attributes:
                dictionary[attribute] = self._population[neurons].get(attribute).tolist()
        else:
            for attribute in attrs:
                dictionary[attribute] = self._population[neurons].get(attribute).tolist()
        return dictionary

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
        Arguments:
            neurons: not implemented yet.
                     was in nest: nest.NodeCollection or sequence (tuple, list, array) of neurons
                     the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            A list of projections.
        """
        self._assert_annarchy()
        # neurons = self._assert_neurons(neurons)
        if source_or_target not in ["source", "target"]:
            return self.projections_in, self.projections_out
        elif source_or_target == "source":
            return self.projections_out
        elif source_or_target == "target":
            return self.projections_in

    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             connections: a projection object.
             values_dict: dictionary of attributes names' and values.
        """
        self._assert_annarchy()
        if connections in self.projections_in or connections in self.projections_out:
            # connections.set(values_dict) <- this would be straightforward, but can generate
            # arbitrary attributes that get ignored by the projection but are readable with get()
            for attr, value in values_dict.items():
                if hasattr(connections, attr):
                    connections.set({attr: value})
                else:
                    raise AttributeError("Projection %s has no attribute named %s." %(connections.name, attr))
        else:
            raise AttributeError("No incoming projection %s associated to this %s of model %s with label %s." %
                             (connections.name, self.__class__.__name__, self.model, self.label))

    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            Returns:
             Dictionary of arrays of connections' attributes.

        """
        self._assert_annarchy()
        dictionary = {}
        if attrs is None:
            for attribute in connections.attributes:
                dictionary[attribute] = connections.get(attribute)
        else:
            for attribute in attrs:
                dictionary[attribute] = connections.get(attribute)
        return dictionary

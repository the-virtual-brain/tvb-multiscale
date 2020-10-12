# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.population import SpikingPopulation

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, extract_integer_intervals, is_integer


class ANNarchyPopulation(SpikingPopulation):

    """ANNarchyPopulation class
       Wraps around an ANNarchy.Population class and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    annarchy_instance = None
    _population = None
    _population_ind = None
    # ANNarchy.Population or PopulationView instances as keys, outgoing projections as values:
    _projections_pre = OrderedDict()   # outgoing pojections
    _projections_post = OrderedDict()  # incoming projections
    _weight_attr = "weights"
    _delay_attr = "delays"
    _receptor_attr = "target"
    _default_connection_attrs = ["pre", "post", _weight_attr, _delay_attr, _receptor_attr]

    def __init__(self, population_neurons, label="", model="", annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        self._population = population_neurons
        if len(label):
            self._population.name = label
        else:
            label = self._population.name
        if annarchy_instance is not None:
            self._population_ind = self._get_population_ind()
        super(ANNarchyPopulation, self).__init__(label, model, **kwargs)

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    def _assert_annarchy(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    @property
    def population(self):
        return self._population

    def _get_population_ind(self):
        return self.annarchy_instance.Global._network[0]["populations"].index(self._population)

    @property
    def population_ind(self):
        self._assert_annarchy()
        if self._population_ind is None:
            self._population_ind = self._get_population_ind()
        return self._population_ind

    @property
    def neurons(self):  # tuple of populations' neurons
        """Method to get all neurons' indices of this population.
           Returns:
            tuple of neurons' global indices.
            In ANNarchy: So far we get only local indices.
            We form global indices by zipping local indices with the global population indice.
        """
        local_ids = self._population.ranks
        return tuple(zip([[self.population_ind] * len(local_ids), local_ids]))

    def _assert_neurons(self, neurons=None):
        """Method to assert an input set of neurons either as:
            - the present instance of ANNarchy.Population class
            - a ANNarchy.PopulationView instance of the present instance of ANNarchy.Population class
            - a collection (tuple, list, numpy.ndarray) of global indices (i.e., tuples of (population_inds, neuron_ind),
              of the present instance of ANNarchy.Population class, or of local indices thereof,
            Default input = None, which corresponds to the present instance of ANNarchy.Population class.
        """
        if neurons is None:
            neurons = self._population
        else:
            if isinstance(neurons, self.annarchy_instance.Population):
                # Assert that we refer to this object's Population
                assert self._population == neurons
            elif isinstance(neurons, self.annarchy_instance.PopulationView):
                # Assert that we refer to a view of this object's Population
                assert self._population == neurons.population
            else:
                # Let's check if these are global or local indices of neurons...
                local_inds = []
                for neuron in ensure_list(neurons):
                    if isinstance(neuron, (tuple, list)):
                        # If neurons are global_ids formed as tuples of (population_ind, neuron_ind)...
                        if neuron[0] == self.population_ind:
                            # ... confirm that the population_ind is correct and get the neuron_ind
                            local_inds.append(neuron[1])
                            # If neurons are just local inds, gather them...
                        elif is_integer(neuron):
                            local_inds.append(neuron)
                        else:
                            raise ValueError(
                                "neurons %s\nis neither an instance of ANNarchy.Population, "
                                "nor of  ANNarchy.PopulationView,\n"
                                "nor is it a collection (tuple, list, or numpy.ndarray) "
                                "of global (tuple of (population_inds, neuron_ind) or local indices of neurons!")
                        # Return a Population View:
                        neurons = self._population[neurons]
        return neurons

    def summarize_neurons_indices(self, print=False):
        """Method to summarize neurons' indices' intervals.
        Arguments:
         print: if True, a string is returned, Default = False
        Returns:
         a list of intervals' limits, or of single indices, or a string of the list if print = True"""
        return extract_integer_intervals([neuron[1] for neuron in self.neurons], print=print)

    def _print_neurons(self):
        """ Prints indices of neurons in this population.
            Currently we get only local indices.
        """
        return "\n%d neurons: %s" % (self.number_of_neurons, self.summarize_neurons_indices(print=True))

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a ANNarchy.Population or ANNarchy.PopulationView class,
                     or a collection (tuple, list, numpy.ndarray) of global indices
                     (i.e., tuples of (population_inds, neuron_ind),
                     of the present instance of ANNarchy.Population class, or of local indices thereof,
        """
        self._assert_annarchy()
        self._assert_neurons(neurons).set(values_dict)

    def _Get(self, attrs=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a ANNarchy.Population or ANNarchy.PopulationView class,
                     or a collection (tuple, list, numpy.ndarray) of global indices
                     (i.e., tuples of (population_inds, neuron_ind),
                     of the present instance of ANNarchy.Population class, or of local indices thereof,
           Returns:
            Dictionary of numpy.arrays of neurons' attributes.
        """
        self._assert_annarchy()
        dictionary = {}
        neurons = self._assert_neurons(neurons)
        if attrs is None:
            # If no attribute is specified, return all of them
            for attribute in neurons.attributes:
                dictionary[attribute] = neurons.get(attribute)
        else:
            for attribute in attrs:
                dictionary[attribute] = neurons.get(attribute)
        return dictionary

    def _get_projections(self, pre_or_post, neurons):
        """Get the projections of this populations.
           Arguments:
            pre_or_post: "pre" or "post" to choose the corresponding connections
            neurons: an ANNarchy.Population or ANNarchy.PopulationView to filter the connections returned
           Return:
            a list of ANNarchy.Projection instances
        """
        neurons = self._assert_neurons(neurons)
        projections = []
        for proj_neurons, proj in getattr(self, "_projections_%s" % pre_or_post).items():
            if proj_neurons.ranks == neurons.ranks:
                projections.append(proj)
        return projections

    @property
    def projections_pre(self):
        return self._projections_pre.keys()

    @property
    def projections_post(self):
        return self._projections_post.keys()

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
        Arguments:
            neurons: ANNarchy.Population or ANNarchy.PopulationView or sequence (tuple, list, array) of neurons
                     the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            A list of Projections' objects.
        """
        self._assert_annarchy()
        neurons = self._assert_neurons(neurons)
        if source_or_target not in ["source", "target"]:
            return self._get_projections("pre", neurons), self._get_projections("post", neurons)
        elif source_or_target == "source":
            return self._get_projections("pre", neurons)
        elif source_or_target == "target":
            return self._get_projections("post", neurons)

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections to/from the present population.
        """
        self._assert_annarchy()
        projections_pre = self._projections_pre
        projections_post = self._projections_post
        if connections is None:
            connections = self._GetConnections()
        for connection in ensure_list(connections):
            if connection in projections_pre or connection in projections_post:
                # connection.set(values_dict) <- this would be straightforward, but can generate
                # arbitrary attributes that get ignored by the projection but are readable with get()
                # TODO: figure out why this is bad, if it doesn't cause an error/Exception!
                for attr, value in values_dict.items():
                    if hasattr(connection, attr):
                        connection.set({attr: value})
                    else:
                        raise AttributeError("Projection %s has no attribute named %s." %(connection.name, attr))
            else:
                raise AttributeError("No incoming projection %s associated to this %s of model %s with label %s." %
                                 (connection.name, self.__class__.__name__, self.model, self.label))

    def _set_attributes_of_connection_to_dict(self, dictionary, connection, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(connection.get(attribute))
        else:
            dictionary[attribute] = [connection.get(attribute)]

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections to/from the present population.
             attrs: collection (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponds to all attributes
            Returns:
             Dictionary of lists (for the possible different Projection objects) of arrays of connections' attributes.
        """
        self._assert_annarchy()
        if connections is None:
            connections = self._GetConnections()
        projections_pre = self.projections_pre
        projections_post = self._projections_post
        outputs = []
        for connection in ensure_list(connections):
            dictionary = {}
            if connection in projections_pre or connection in projections_post:
                if attrs is None:
                    for attribute in np.union1d(self._default_connection_attrs,
                                                connection.attributes):
                        self._set_attributes_of_connection_to_dict(dictionary, connection, attribute)
                else:
                    for attribute in attrs:
                        self._set_attributes_of_connection_to_dict(dictionary, connection, attribute)
            outputs.append(dictionary)
        if len(outputs) == 0:
            return {}
        elif len(outputs) == 1:
            # for source or target connections
            return outputs[0]
        else:
            # for source and target connections
            return tuple(outputs)

    def get_number_of_neurons(self):
        """Method to compute the total number of ANNarchyPopulation's neurons.
            Returns:
                int: number of neurons.
        """
        return self._population.size
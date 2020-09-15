# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_multiscale.core.config import initialize_logger

from tvb_multiscale.core.utils.data_structures_utils import filter_neurons, summarize

from tvb.contrib.scripts.utils.data_structures_utils import \
    flatten_tuple, extract_integer_intervals, list_of_dicts_to_dict_of_lists


LOG = initialize_logger(__name__)


class SpikingPopulation(object):
    __metaclass__ = ABCMeta

    """This is a class wraping around the neurons of a spiking population."""

    label = ""    # label of population
    model = ""    # label of neuronal model
    _number_of_neurons = 0  # total number of populations' neurons

    # Default attributes' labels:
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    @property
    @abstractmethod
    def neurons(self):  # tuple of populations' neurons
        """Method to get all neurons' indices of this population.
           Returns:
            tuple of neurons'indices.
        """
    pass

    def __init__(self, label="", model="", *args, **kwargs):
        self.label = str(label)
        self.model = str(model)
        self._number_of_neurons = self.number_of_neurons

    def __setitem__(self, keys, values):
        """Slice specific neurons (keys) of this SpikingPopulation.
           Argument:
            keys: collection of target populations' keys.
           Returns:
           Sub-tuple of SpikingPopulation.neurons.
        """
        neurons = np.array(self.neurons)
        neurons[keys] = values
        self.neurons = tuple(neurons)

    def __getitem__(self, keys):
        """Slice specific neurons (keys) of this SpikingPopulation.
           Argument:
            keys: collection of target populations' keys.
           Returns:
           Sub-tuple of SpikingPopulation.neurons.
        """
        return tuple(np.array(self.neurons)[keys])

    def summarize_neurons_indices(self, print=False):
        """Method to summarize neurons' indices' intervals.
        Arguments:
         print: if True, a string is returned, Default = False
        Returns:
         a list of intervals' limits, or of single indices, or a string of the list if print = True"""
        return extract_integer_intervals(self.neurons, print=print)

    def __repr__(self):
        return "%s - Label: %s, %d neurons: %s" % (self.__class__.__name__, self.label,
                                                   self.number_of_neurons, self.summarize_neurons_indices(print=True))

    def _print_neurons(self):
        return "\n%d neurons: %s" % (self.number_of_neurons, self.summarize_neurons_indices(print=True))

    def __str__(self):
        return "\n%s - Label: %s" \
               "%s" \
               "\nparameters: %s," % \
                          (self.__class__.__name__,  self.label,
                           self._print_neurons(),
                           str(self.get_attributes(summary=True)))

    def print_str(self, connectivity=False):
        output = self.__str__()
        if connectivity is True:
            conn_attrs = self.GetFromConnections(attrs=[self._weight_attr, self._delay_attr, self._receptor_attr],
                                                 summary=3)
            output += "\nconnections' weights: %s, \nconnections' delays: %s,\nconnections' receptors: %s" % \
                      (str(conn_attrs.get(self._weight_attr, "")),
                       str(conn_attrs.get(self._delay_attr, "")),
                       str(conn_attrs.get(self._receptor_attr, "")))
        return output

    # Methods to get or set attributes for neurons and/or their connections:

    @abstractmethod
    def _Set(self, neurons, values_dict):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            neurons: tuple of neurons the attributes of which should be set.
            values_dict: dictionary of attributes names' and values.
        """
        pass

    @abstractmethod
    def _Get(self, neurons, attr=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            neurons: tuple of neurons which should be included in the output.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of lists of neurons' attributes.
        """
        pass

    @abstractmethod
    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
           Arguments:
            neurons: tuple of neurons the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        pass

    @abstractmethod
    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             connections: connections' objects.
             values_dict: dictionary of attributes names' and values.
        """
        pass

    @abstractmethod
    def _GetFromConnections(self, connections, attr=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            Returns:
             Dictionary of lists of connections' attributes.

        """
        pass

    def filter_neurons(self, neurons=None, exclude_neurons=[]):
        """This method will select/exclude the connected neurons, depending on user inputs
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            tuple of neurons.
        """
        if neurons is None:
            neurons = self.neurons
        if len(neurons) == 0:
            return ()
        return filter_neurons(neurons, exclude_neurons)

    def get_number_of_neurons(self, neurons=None, exclude_neurons=[]):
        """Method to compute the total number of SpikingPopulation's neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            Returns:
                int: number of neurons.
        """
        return len(self.filter_neurons(neurons, exclude_neurons))

    def Set(self, values_dict, neurons=None, exclude_neurons=[]):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: collection (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
        """
        return self._Set(self.filter_neurons(neurons, exclude_neurons), values_dict)

    def Get(self, attrs=None, neurons=None, exclude_neurons=[], summary=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of lists of neurons' attributes.
        """
        attributes = self._Get(self.filter_neurons(neurons, exclude_neurons), attrs)
        if summary:
            return summarize(attributes, summary)
        else:
            return attributes

    def get_attributes(self, neurons=None, exclude_neurons=[], summary=False):
        """Method to get all attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of lists of neurons' attributes.
        """
        return self.Get(neurons=neurons, exclude_neurons=exclude_neurons, summary=summary)

    def GetConnections(self, neurons=None, exclude_neurons=[], source_or_target=None):
        """Method to get all connections of the device to/from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons
                             which should be excluded from the output. Default = [].
            Returns:
                connections' objects.
        """
        return self._GetConnections(self.filter_neurons(neurons, exclude_neurons), source_or_target)

    def SetToConnections(self, values_dict, neurons=None, exclude_neurons=[], source_or_target=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: collection (list, tuple, array) of neurons the attribute of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
        """
        self._SetToConnections(flatten_tuple(self.GetConnections(neurons, exclude_neurons, source_or_target)),
                               values_dict)

    def GetFromConnections(self, attrs=None, neurons=None, exclude_neurons=[], source_or_target=None,
                           summary=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of lists of connections' attributes.
        """
        connections = self.GetConnections(neurons, exclude_neurons, source_or_target)
        output = []
        for conn in connections:
            if summary is not None:
                output.append(summarize(self._GetFromConnections(conn, attrs), summary))
            else:
                output.append(self._GetFromConnections(conn, attrs))
        return list_of_dicts_to_dict_of_lists(output)

    def get_weights(self, neurons=None, exclude_neurons=[], source_or_target=None, summary=None):
        """Method to get the connections' weights of the SpikingPopulations's neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of population's connections' weights.
        """
        return self.GetFromConnections(self._weight_attr, neurons, exclude_neurons,
                                       source_or_target, summary)[self._weight_attr]

    def get_delays(self, neurons=None, exclude_neurons=[], source_or_target=None, summary=None):
        """Method to get the connections' delays of the SpikingPopulations's neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of population's connections' delays.
        """
        return self.GetFromConnections(self._delay_attr, neurons, exclude_neurons,
                                       source_or_target, summary)[self._delay_attr]

    def get_receptors(self, neurons=None, exclude_neurons=[], source_or_target=None, summary=None):
        """Method to get the connections' receptors of the SpikingPopulations's neurons.
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of population's connections' receptors.
        """
        return self.GetFromConnections(self._receptor_attr, neurons, exclude_neurons,
                                       source_or_target, summary)[self._receptor_str]

    @property
    def number_of_neurons(self):
        """Method to get the total number of SpikingPopulation's neurons and set the respective protected property.
            Returns:
             int: number of neurons.
        """
        if self._number_of_neurons == 0 or self._number_of_neurons is None:
            self._number_of_neurons = self.get_number_of_neurons()
        return self._number_of_neurons

    @property
    def attributes(self):
        """Method to get the attributes of the SpikingPopulation's neurons.
           Returns:
            Dictionary of lists of population's neurons' attributes.
        """
        return self.get_attributes()

    @property
    def connections(self):
        """Method to get the connections of the SpikingPopulation's neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections()

    @property
    def weights(self):
        """Method to get the connections' weights' statistical summary of the SpikingPopulations's neurons.
           Returns:
            Arrays of population's connections' weights.
        """
        return self.get_weights()

    @property
    def delays(self):
        """Method to get the connections' delays of the SpikingPopulations's neurons.
           Returns:
            Arrays of population's connections' delays.
        """
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get the connections' receptors of the SpikingPopulations's neurons.
           Returns:
            Arrays of population's connections' receptors.
        """
        return self.get_receptors()

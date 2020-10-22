# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from tvb_multiscale.core.config import initialize_logger

from tvb_multiscale.core.utils.data_structures_utils import summarize


LOG = initialize_logger(__name__)


class SpikingPopulation(object):
    __metaclass__ = ABCMeta

    """SpikingPopulation is a class that 
       represents a population of spiking neurons of the same neural model, 
       residing at the same brain region.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
    """

    _population = None  # Class instance of a sequence of neurons, that depends on its spiking simulator
    label = ""    # label of population
    model = ""    # label of neuronal model
    _number_of_neurons = 0  # total number of populations' neurons

    # Default attributes' labels:
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, population=None, label="", model=""):
        """Constructor of a population class.
           Arguments:
            population: Class instance of a sequence of neurons, that depends on its spiking simulator. Default=None.
            label: a string with the label of the population
            model: a string with the name of the model of the population
        """
        self._population = population
        self.label = str(label)
        self.model = str(model)
        self._number_of_neurons = self.number_of_neurons

    def __getitem__(self, keys):
        """Slice specific neurons (keys) of this SpikingPopulation.
           Argument:
            keys: sequence of target populations' keys.
           Returns:
            Sub-collection of SpikingPopulation.neurons.
        """
        return self._population[keys]

    @abstractmethod
    def _print_neurons(self):
        pass

    def __repr__(self):
        return "%s - Label: %s \n%d neurons: %s" % (self.__class__.__name__, self.label,
                                                    self.number_of_neurons, self._print_neurons())

    def __str__(self):
        return "\n%s" \
               "\nparameters: %s," % \
                          (self.__repr__(), str(self.get_attributes(summary=True)))

    def print_str(self, connectivity=False):
        output = self.__str__()
        if connectivity is True:
            conn_attrs = self.GetFromConnections(attrs=[self._weight_attr, self._delay_attr, self._receptor_attr],
                                                 source_or_target="source", summary=3)
            output += "\nconnections from %s:\nweights: %s,\ndelays: %s,\nreceptors: %s" % \
                      (self.label,
                       str(conn_attrs.get(self._weight_attr, "")),
                       str(conn_attrs.get(self._delay_attr, "")),
                       str(conn_attrs.get(self._receptor_attr, "")))
            conn_attrs = self.GetFromConnections(attrs=[self._weight_attr, self._delay_attr, self._receptor_attr],
                                                 source_or_target="target", summary=3)
            output += "\nconnections to %s:\nweights: %s,\ndelays: %s,\nreceptors: %s" % \
                      (self.label,
                       str(conn_attrs.get(self._weight_attr, "")),
                       str(conn_attrs.get(self._delay_attr, "")),
                       str(conn_attrs.get(self._receptor_attr, "")))
        return output

    # Methods to get or set attributes for neurons and/or their connections:

    @property
    @abstractmethod
    def neurons(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of populations' neurons"""
        pass

    @abstractmethod
    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        pass

    @abstractmethod
    def _Get(self, attr=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of neurons' attributes.
        """
        pass

    @abstractmethod
    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        pass

    @abstractmethod
    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
        """
        pass

    @abstractmethod
    def _GetFromConnections(self, attr=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponding to all attributes
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
            Returns:
             Dictionary of sequences (lists, tuples, or arrays) of connections' attributes.

        """
        pass

    def get_number_of_neurons(self):
        """Method to compute the total number of SpikingPopulation's neurons.
            Returns:
                int: number of neurons.
        """
        return len(self._population)

    def Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        self._Set(values_dict, neurons)

    def Get(self, attrs=None, neurons=None, summary=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of neurons' attributes.
        """
        attributes = self._Get(attrs, neurons)
        if summary:
            return summarize(attributes, summary)
        else:
            return attributes

    def get_attributes(self, neurons=None, summary=False):
        """Method to get all attributes of the SpikingPopulation's neurons.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of neurons' attributes.
        """
        return self.Get(neurons=neurons, summary=summary)

    def GetConnections(self, neurons=None,  source_or_target=None):
        """Method to get all connections of the device to/from neurons.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            Returns:
                connections' objects.
        """
        return self._GetConnections(neurons, source_or_target)

    def SetToConnections(self, values_dict, neurons=None, source_or_target=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
        """
        if source_or_target is None:
            # In case we deal with both source and target connections, treat them separately:
            for source_or_target in ["source", "target"]:
                self.SetToConnections(values_dict, neurons, source_or_target)
        self._SetToConnections(values_dict, self.GetConnections(neurons, source_or_target))

    def GetFromConnections(self, attrs=None, neurons=None, source_or_target=None, summary=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, correspondingn to all attributes
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
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
        if source_or_target is None:
            # In case we deal with both source and target connections, treat them separately:
            output = []
            for source_or_target in ["source", "target"]:
                output.append(self.GetFromConnections(attrs, neurons, source_or_target, summary))
            return tuple(output)
        outputs = self._GetFromConnections(attrs, self.GetConnections(neurons, source_or_target))
        if summary is not None:
            outputs = summarize(outputs, summary)
        return outputs

    def get_weights(self, neurons=None, source_or_target=None, summary=None):
        """Method to get the connections' weights of the SpikingPopulations's neurons.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of population's connections' weights.
        """
        if source_or_target is None:
            # In case we deal with both source and target connections, treat them separately:
            outputs = []
            for source_or_target in ["source", "target"]:
                outputs.append(self.get_weights(neurons, source_or_target, summary))
            return tuple(outputs)
        return self.GetFromConnections(self._weight_attr, neurons, source_or_target, summary)[self._weight_attr]

    def get_delays(self, neurons=None, source_or_target=None, summary=None):
        """Method to get the connections' delays of the SpikingPopulations's neurons.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of population's connections' delays.
        """
        if source_or_target is None:
            # In case we deal with both source and target connections, treat them separately:
            outputs = []
            for source_or_target in ["source", "target"]:
                outputs.append(self.get_delays(neurons, source_or_target, summary))
            return tuple(outputs)
        return self.GetFromConnections(self._delay_attr, neurons, source_or_target, summary)[self._delay_attr]

    def get_receptors(self, neurons=None, source_or_target=None, summary=None):
        """Method to get the connections' receptors of the SpikingPopulations's neurons.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of population's connections' receptors.
        """
        if source_or_target is None:
            # In case we deal with both source and target connections, treat them separately:
            outputs = []
            for source_or_target in ["source", "target"]:
                outputs.append(self.get_receptors(neurons, source_or_target, summary))
            return tuple(outputs)
        return self.GetFromConnections(self._receptor_attr, neurons, source_or_target, summary)[self._receptor_str]

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
            Dictionary of sequences (lists, tuples, or arrays)  of population's neurons' attributes.
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
            Dictionary of sequences (lists, tuples, or arrays) of population's connections' weights.
        """
        return self.get_weights()

    @property
    def delays(self):
        """Method to get the connections' delays of the SpikingPopulations's neurons.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of population's connections' delays.
        """
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get the connections' receptors of the SpikingPopulations's neurons.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of population's connections' receptors.
        """
        return self.get_receptors()

# -*- coding: utf-8 -*-
from six import string_types
from pandas import Series

from tvb_multiscale.config import initialize_logger, LINE

from tvb.contrib.scripts.utils.data_structures_utils import series_loop_generator, is_integer


LOG = initialize_logger(__name__)


class SpikingRegionNode(Series):

    """This is an indexed mapping between populations labels and
       the neuronal populations residing at a specific brain region node,
       based on inheriting pandas.Series class"""

    _number_of_neurons = 0

    # Default attributes' labels:
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, label="", input_nodes=None, **kwargs):
        super(SpikingRegionNode, self).__init__(input_nodes, name=label, **kwargs)
        self._number_of_neurons = self.number_of_neurons

    def __str__(self):
        populations = ""
        for pop in self.populations:
            populations += str(self[pop])
        return LINE + "Node Label: %s\n Populations: %s" % (self.label, populations)

    def __len__(self):
        return super(SpikingRegionNode, self).__len__()

    def __getitem__(self, items):
        if isinstance(items, string_types) or is_integer(items):
            return super(SpikingRegionNode, self).__getitem__(items)
        return SpikingRegionNode(label=self.label, input_nodes=super(SpikingRegionNode, self).__getitem__(items))

    @property
    def label(self):
        """The region node label."""
        return super(SpikingRegionNode, self).name

    # Methods to get or set attributes for neurons and/or their connections:

    def _loop_generator(self, pop_inds_or_lbls=None):
        """Method to create a generator looping through the SpikingPopulation objects
         and returning the indice, the label, and the SpikingPopulation itself.
            Arguments:
             pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                               Default = None, corresponds to all populations of the region node.
            Returns:
             The generator object
         """
        return series_loop_generator(self, pop_inds_or_lbls)

    def get_neurons(self, pop_inds_or_lbls=None):
        """Method to get the neurons of the SpikingRegionNode's populations.
           Argument:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
           Returns:
            tuple of neurons."""
        output = ()
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            output += tuple(pop.neurons)
        return output

    def get_number_of_neurons(self, pop_inds_or_lbls=None):
        return len(self.get_neurons(pop_inds_or_lbls))

    def Set(self, values_dict, pop_inds_or_lbls=None):
        """Method to set attributes of the SpikingRegionNode's populations' neurons.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the region node.
        """
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            pop.Set(values_dict)

    def Get(self, attrs=None, pop_inds_or_lbls=None, summary=False):
        """Method to get attributes of the SpikingRegionNode's populations' neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                             Default = None, corresponds to all populations of the region node.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of arrays of populations' neurons' attributes.
        """
        output = Series()
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            output[lbl] = pop.Get(attrs, summary=summary)
        return output

    def get_attributes(self, pop_inds_or_lbls=None, summary=False):
        """Method to get all attributes of the SpikingRegionNode's populations' neurons.
           Arguments:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                             Default = None, corresponds to all populations of the region node.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of arrays of populations' neurons' attributes.
        """
        return self.Get(pop_inds_or_lbls=pop_inds_or_lbls, summary=summary)

    def GetConnections(self, pop_inds_or_lbls=None, source_or_target=None):
        """Method to get the connections of the SpikingRegionNode's populations.
           Argument:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            Series of connections.
        """
        output = Series()
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            output[lbl] = pop.GetConnections(source_or_target=source_or_target)
        return output

    def SetToConnections(self, values_dict, pop_inds_or_lbls=None, source_or_target=None):
        """Method to set attributes of the connections from/to the SpikingRegionNode's populations.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
        """
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            pop.SetToConnections(values_dict, source_or_target=source_or_target)

    def GetFromConnections(self, attrs=None, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get attributes of the connections from/to the SpikingRegionNode's populations.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of arrays of connections' attributes.
        """
        output = Series()
        for id, lbl, pop in self._loop_generator(pop_inds_or_lbls):
            output[lbl] = pop.GetFromConnections(attrs, source_or_target=source_or_target, summary=summary)
        return output

    def get_weights(self, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' weights of the SpikingRegionNode's neurons.
           Argument:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of populations' neurons' weights.
        """
        return self.GetFromConnections(self._weight_attr, pop_inds_or_lbls, source_or_target, summary)

    def get_delays(self, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' delays of the SpikingRegionNode's neurons.
           Argument:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of populations' neurons' delays.
        """
        return self.GetFromConnections(self._delay_attr, pop_inds_or_lbls, source_or_target, summary)

    def get_receptors(self, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' receptors of the SpikingRegionNode's neurons.
           Argument:
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                               Default = None, corresponds to all populations of the SpikingRegionNode.
            source_or_target: Direction of connections relative to the region's neurons
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of populations' neurons' receptors.
        """
        return self.GetFromConnections(self._receptor_attr, pop_inds_or_lbls, source_or_target, summary)

    @property
    def node(self):
        """Method to return self (SpikingRegionNode)."""
        return self

    @property
    def populations(self):
        """Method to get the list of populations' labels of the SpikingRegionNode.
           Returns:
            list of populations' labels.
        """
        return list(self.index)

    @property
    def neurons(self):
        """Method to get the neurons of the SpikingRegionNode's populations.
           Returns:
            tuple of neurons."""
        return self.get_neurons()

    @property
    def number_of_neurons(self):
        """Method to get the total number of neurons of the SpikingRegionNode's populations,
           and for setting the respective protected property."""
        if self._number_of_neurons is None or self._number_of_neurons == 0:
            self._number_of_neurons = self.get_number_of_neurons()
        return self._number_of_neurons

    @property
    def attributes(self):
        return self.get_attributes()

    @property
    def connections(self):
        """Method to get the connections of the SpikingRegionNode's neurons.
           Returns:
            Series of connections.
        """
        return self.GetConnections()

    @property
    def weights(self):
        """Method to get the connections' weights of the SpikingRegionNode's neurons.
           Returns:
            Series of populations' neurons' weights.
        """
        return self.get_weights()

    @property
    def delays(self):
        """Method to get the connections' delays of the SpikingRegionNode's neurons.
           Returns:
            Series of populations' neurons' delays.
        """
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get the connections' receptors of the SpikingRegionNode's neurons.
           Returns:
            Series of populations' neurons' receptors.
        """
        return self.get_receptors()

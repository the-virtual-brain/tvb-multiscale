# -*- coding: utf-8 -*-

from six import string_types
from collections import OrderedDict

import numpy as np
from pandas import Series

from tvb_multiscale.core.config import initialize_logger, LINE

from tvb.basic.neotraits.api import HasTraits, Int

from tvb.contrib.scripts.utils.data_structures_utils import \
    series_loop_generator, is_integer, concatenate_heterogeneous_DataArrays


LOG = initialize_logger(__name__)


class SpikingBrain(Series, HasTraits):

    """SpikingBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective SpikingRegionNode instances.
    """

    _number_of_neurons = Int(field_type=int, default=0, required=True, label="Number of neurons",
                             doc="""The number of neurons of SpikingBrain""")

    # # Default attributes' labels:
    # _weight_attr = "weight"
    # _delay_attr = "delay"
    # _receptor_attr = "receptor"
    
    def __init__(self, input_brain=None, **kwargs):
        Series.__init__(self, input_brain, **kwargs)
        HasTraits.__init__(self)
        self._number_of_neurons = self.get_number_of_neurons()

    def __len__(self):
        return super(SpikingBrain, self).__len__()

    def __repr__(self):
        return "\nSpikingBrain - Regions: %s" % str(self.regions)

    def __str__(self):
        return self.print_str()

    def print_str(self, connectivity=False):
        output = self.__repr__() + "\nRegions' nodes:\n"
        for region in self.regions:
            output += "\n" + LINE + self[region].print_str(connectivity)
        return output

    def __getitem__(self, items):
        if isinstance(items, string_types) or is_integer(items):
            return super(SpikingBrain, self).__getitem__(items)
        return SpikingBrain(input_brain=super(SpikingBrain, self).__getitem__(items))

    def _loop_generator(self, reg_inds_or_lbls=None):
        """Method to create a generator looping through the SpikingBrain's SpikingRegionNode objects
         and returning the indice, the label, and the SpikingRegionNode itself.
            Arguments:
             reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                               Default = None, corresponds to all populations of the region node.
            Returns:
             The generator object
         """
        return series_loop_generator(self, reg_inds_or_lbls)

    def get_neurons(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
        """Method to get the neurons of the SpikingBrain.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
           Returns:
            tuple of neurons.
        """
        output = ()
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output += tuple(reg.get_neurons(pop_inds_or_lbls))
        return tuple(output)

    def get_number_of_neurons(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
        """Method to get the number of neurons of the SpikingBrain.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
           Returns:
            int: number of neurons.
        """
        return len(self.get_neurons(reg_inds_or_lbls, pop_inds_or_lbls))

    def get_number_of_neurons_per_region(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, fill_value=0):
        output = Series()
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.get_number_of_neurons_per_population(pop_inds_or_lbls)
        return concatenate_heterogeneous_DataArrays(output, concat_dim_name="Region",
                                                    name="Number of neurons", fill_value=fill_value)

    def get_populations_sizes(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
        pops_per_region = self.get_number_of_neurons_per_region(reg_inds_or_lbls, pop_inds_or_lbls, fill_value=np.nan)
        populations = pops_per_region.coords["Population"]
        populations_sizes = OrderedDict()
        for pop in populations.values.tolist():
            populations_sizes[pop] = np.nanmean(pops_per_region.loc[:, pop].values.squeeze()).item()
        return populations_sizes

    @property
    def populations_sizes(self):
        return self.get_populations_sizes()

    def Set(self, values_dict, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
        """Method to set attributes of the SpikingBrain's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
        """
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            reg.set(values_dict, pop_inds_or_lbls)

    def Get(self, attrs=None, reg_inds_or_lbls=None, pop_inds_or_lbls=None, summary=None):
        """Method to get attributes of the SpikingBrain's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of populations' neurons' attributes.
        """
        output = Series()
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.get(attrs, pop_inds_or_lbls, summary)
        return output

    def get_attributes(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, summary=None):
        """Method to get all attributes of the SpikingBrain's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of populations' neurons' attributes.
        """
        return self.Get(reg_inds_or_lbls=reg_inds_or_lbls, pop_inds_or_lbls=pop_inds_or_lbls, summary=summary)

    def GetConnections(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None):
        """Method to get the connections of the SpikingBrain's populations.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            Returns:
             Series of region Series of connections.
        """
        output = Series()
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.GetConnections(pop_inds_or_lbls, source_or_target)
        return output

    def SetToConnections(self, values_dict, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None):
        """Method to set attributes of the connections from/to the SpikingBrain's populations.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
        """
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            reg.SetToConnections(values_dict, pop_inds_or_lbls, source_or_target)

    def GetFromConnections(self, attrs=None, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None,
                           summary=None):
        """Method to get attributes of the connections from/to the SpikingBrain's populations.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Series of region Series of connections' attributes.
        """
        output = Series()
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.GetFromConnections(attrs, pop_inds_or_lbls, source_or_target, summary)
        return output

    def get_weights(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' weights of the SpikingBrain's neurons.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            All brains' connections' weights organized in nested Series of regions and populations.
        """
        return self.GetFromConnections(self._weight_attr, reg_inds_or_lbls, pop_inds_or_lbls, source_or_target, summary)

    def get_delays(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' delays of the SpikingBrain's neurons.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            All brains' connections' delays organized in nested Series of regions and populations.
        """
        return self.GetFromConnections(self._delay_attr, reg_inds_or_lbls, pop_inds_or_lbls, source_or_target, summary)

    def get_receptors(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' receptors of the SpikingBrain's neurons.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            All brains' connections' receptors organized in nested Series of regions and populations.
        """
        return self.GetFromConnections(self._receptor_attr, reg_inds_or_lbls, pop_inds_or_lbls, source_or_target,
                                       summary)

    @property
    def regions(self):
        """Method to get all regions' labels of the SpikingBrain
           Returns:
               list: regions' labels.
        """
        return list(self.index)

    @property
    def neurons(self):
        """Method to get all the neurons of the SpikingBrain.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
           Returns:
            tuple of neurons.
        """
        return self.get_neurons()

    @property
    def number_of_neurons(self):
        """Method to get the total number of neurons of the SpikingBrain
           and for setting the respective protected property.
           Returns:
            int: number of neurons.
        """
        if self._number_of_neurons is None or self._number_of_neurons == 0:
            self._number_of_neurons = self.get_number_of_neurons()
        return self._number_of_neurons

    @property
    def number_of_neurons_per_region(self):
        return self.get_number_of_neurons_per_region()

    @property
    def attributes(self):
        """Method to get all attributes of the SpikingBrain's neurons.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
           Returns:
            Series of populations' neurons' attributes.
        """
        return self.get_attributes()

    @property
    def connections(self):
        """Method to get the connections of the SpikingBrain's neurons.
           Returns:
            All brains' connections organized in nested Series of regions and populations.
        """
        return self.GetConnections()

    @property
    def weights(self):
        """Method to get the connections' weights of the SpikingBrain's neurons.
           Returns:
            Series of populations' neurons' weights.
        """
        return self.get_weights()

    @property
    def delays(self):
        """Method to get the connections' delays of the SpikingBrain's neurons.
           Returns:
            Series of populations' neurons' delays.
        """
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get the connections' receptors of the SpikingBrain's neurons.
           Returns:
            Series of populations' neurons' receptors.
        """
        return self.get_receptors()

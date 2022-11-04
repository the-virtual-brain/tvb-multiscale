# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
from pandas import Series

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.node_set import SpikingNodesSet

from tvb.contrib.scripts.utils.data_structures_utils import \
    series_loop_generator, is_integer, concatenate_heterogeneous_DataArrays


LOG = initialize_logger(__name__)


class SpikingBrain(SpikingNodesSet):

    """SpikingBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and the respective SpikingRegionNode instances.
    """

    _number_of_neurons = None

    _weight_attr = ""
    _delay_attr = ""
    _receptor_attr = ""

    _collection_name = "Region"

    def __getstate__(self):
        return super(SpikingBrain, self).__getstate__()

    def __setstate__(self, d):
        super(SpikingBrain, self).__setstate__(d)

    def get_number_of_neurons_per_region(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, fill_value=0):
        output = Series(dtype='object')
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.get_number_of_neurons_per_population(pop_inds_or_lbls)
        return concatenate_heterogeneous_DataArrays(output, concat_dim_name=self._collection_name,
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
            reg.Set(values_dict, pop_inds_or_lbls)

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
        output = Series(dtype='object')
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.Get(attrs, pop_inds_or_lbls, summary)
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
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
            source_or_target: Direction of connections relative to each neuron
                              "source", "target" or None (Default; corresponds to both source and target)
            Returns:
             Series of region Series of connections.
        """
        output = Series(dtype='object')
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
        output = Series(dtype='object')
        for id, lbl, reg in self._loop_generator(reg_inds_or_lbls):
            output[lbl] = reg.GetFromConnections(attrs, pop_inds_or_lbls, source_or_target, summary)
        return output

    def get_weights(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None, source_or_target=None, summary=None):
        """Method to get the connections' weights of the SpikingBrain's neurons.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
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
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
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
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
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

    def get_neurons(self, reg_inds_or_lbls=None, inds_or_lbls=None):
        """Method to get all the neurons of the SpikingBrain.
           Argument:
            reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
                              Default = None, corresponds to all regions of the SpikingBrain.
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                              Default = None, corresponds to all populations of each SpikingRegionNode.
           Returns:
            tuple of neurons.
        """
        output = ()
        for id, lbl, nodes in self._loop_generator(reg_inds_or_lbls):
            output += tuple(nodes.get_nodes(inds_or_lbls))
        return output

    @property
    def neurons(self):
        """Method to get all the neurons of the SpikingBrain.
           Returns:
            tuple of neurons.
        """
        return self.get_neurons()

    @property
    def number_of_neurons_per_region(self):
        return self.get_number_of_neurons_per_region()

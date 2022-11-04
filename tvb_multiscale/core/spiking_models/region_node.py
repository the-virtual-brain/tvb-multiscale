# -*- coding: utf-8 -*-

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.node_set import SpikingNodesSet


LOG = initialize_logger(__name__)


class SpikingRegionNode(SpikingNodesSet):

    """SpikingRegionNode class is an indexed mapping
       (based on inheriting from pandas.Series class)
       between populations labels and the neuronal populations
       residing at a specific brain region node.
    """

    _number_of_nodes = None

    _collection_name = "Population"

    _weight_attr = ""
    _delay_attr = ""
    _receptor_attr = ""

    # Methods to get or set attributes for neurons and/or their connections:

    def __getstate__(self):
        return super(SpikingRegionNode, self).__getstate__()

    def __setstate__(self, d):
        super(SpikingRegionNode, self).__setstate__(d)

    @property
    def label(self):
        label = self.name
        for pop in self.collections:
            label = self[pop].brain_region
        return label

    def get_neurons(self, inds_or_lbls=None):
        """Method to get the neurons indices of the SpikingRegionNode's populations.
           Argument:
            inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
                          Default = None, corresponds to all populations of the SpikingRegionNode.
           Returns:
            tuple of neurons."""
        return self.get_nodes(inds_or_lbls)

    def get_number_of_neurons(self, inds_or_lbls=None):
        return self.get_number_of_nodes(inds_or_lbls)

    def get_number_of_neurons_per_population(self, inds_or_lbls=None):
        return self.get_number_of_nodes_per_collection(inds_or_lbls)

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
        return self.get_nodes()

    @property
    def number_of_neurons(self):
        """Method to get the total number of neurons of the SpikingRegionNode's populations,
           and for setting the respective protected property."""
        return self.number_of_nodes

    @property
    def number_of_neurons_per_population(self):
        return self.get_number_of_nodes_per_collection()

# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.node import SpikingNodeCollection
from tvb.basic.neotraits.api import Attr, Int


LOG = initialize_logger(__name__)


class SpikingPopulation(SpikingNodeCollection):
    __metaclass__ = ABCMeta

    """SpikingPopulation is a class that 
       represents a population of spiking neurons of the same neural model, 
       residing at the same brain region.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
    """

    label = Attr(field_type=str, default="", required=True,
                 label="Population label", doc="""Label of SpikingPopulation""")

    model = Attr(field_type=str, default="", required=True, label="Population model",
                 doc="""Label of neuronal model of SpikingPopulation's neurons""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the spiking population resides""")

    _size = Int(field_type=int, default=0, required=True, label="Size",
                doc="""The number of neurons of SpikingPopulation """)

    def __getstate__(self):
        return super(SpikingPopulation, self).__getstate__()

    def __setstate__(self, d):
        super(SpikingPopulation, self).__setstate__(d)

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
    def population(self):
        return self._nodes

    @property
    def neurons(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of populations' neurons"""
        return self.gids

    def _assert_neurons(self, neurons=None):
        return self._assert_nodes(neurons)

    def get_number_of_neurons(self):
        """Method to get the number of  neurons connected to/from the device.
           Returns:
            int: number of connections
        """
        return self.get_size()

    @property
    def number_of_neurons(self):
        """Method to get the total number of SpikingPopulation's neurons and set the respective protected property.
            Returns:
             int: number of neurons.
        """
        if self._size == 0 or self._size is None:
            self._size = self.get_size()
        return self._size

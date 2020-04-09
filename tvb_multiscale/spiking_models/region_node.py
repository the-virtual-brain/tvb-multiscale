# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series

from tvb.simulator.plot.utils.log_error_utils import initialize_logger
from tvb.simulator.plot.utils.data_structures_utils import flatten_tuple


LOG = initialize_logger(__name__)


class SpikingRegionNode(Series):
    __metaclass__ = ABCMeta

    # This is an indexed mapping between populations labels and the
    # indices of neurons belonging to the corresponding populations
    label = ""

    def __init__(self, label="", input_node=Series()):
        self.label = str(label)
        super(SpikingRegionNode, self).__init__(input_node)

    def __getitem__(self, keys):
        # return the neurons' indices/handles of specific populations (keys) of this RegionNode
        return flatten_tuple(super(SpikingRegionNode, self).__getitem__(keys))

    # Methods to get or set methods for devices or their connections:

    @abstractmethod
    def Get(self, params=None, indices_or_keys=None):
        pass

    @abstractmethod
    def Set(self, values_dict, indices_or_keys=None):
        pass

    @abstractmethod
    def _get_connections(self, neuron):
        pass

    @abstractmethod
    def GetFromConnections(self, connections, attr=None):
        pass

    @abstractmethod
    def SetToConnections(self, connections, values_dict):
        pass

    @property
    def node(self):
        return self

    @property
    def populations(self):
        return list(self.index)

    def neurons(self, indices_or_keys=None):
        # Return the neurons of this region...
        if indices_or_keys is None:
            # ...either of all populations...
            return flatten_tuple(self)
        else:
            # ...or of selected ones:
            return self.__getitem__(indices_or_keys)

    @property
    def connections(self, indices_or_keys=None):
        # Return the neurons of this region...
        # ...either of all populations...
        # ...or of selected ones:
        connections = []
        for neuron in self.neurons(indices_or_keys):
            connections.append(self._get_connections(neuron))
        return tuple(connections)

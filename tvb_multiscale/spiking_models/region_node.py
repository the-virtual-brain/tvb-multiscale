# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series, unique

import numpy as np

from tvb_multiscale.config import initialize_logger, LINE

from tvb.contrib.scripts.utils.data_structures_utils import flatten_tuple, extract_integer_intervals


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

    def __str__(self):
        populations = ""
        for pop in self.populations:
            neurons = self.neurons(pop)
            populations += LINE + \
                "Label: %s, %d neurons: %s" \
                "\nunique connections' weights: %s, " \
                "\nunique connections' delays: %s, " \
                "\nunique connections' receptors: %s" % \
                           (pop, len(neurons), extract_integer_intervals(neurons, print=True),
                 str(self.get_node_weight(pop)),
                 str(self.get_node_delay(pop)),
                 str(self.get_node_receptors(pop)))
        return "Node Label: %s\n Populations: %s" % (self.label, populations)

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
    def number_of_neurons(self):
        return len(self.neurons())

    def get_connections(self, indices_or_keys=None):
        # Return the neurons of this region...
        # ...either of all populations...
        # ...or of selected ones:
        connections = []
        for neuron in self.neurons(indices_or_keys):
            connections.append(self._get_connections(neuron))
        return tuple(connections)

    @property
    def connections(self):
        return self.get_connections()

    def get_weights(self, pop=None):
        return np.array([self.GetFromConnections(conn, "weight")
                         for conn in self.get_connections(pop)]).flatten()

    def get_delays(self, pop=None):
        return np.array([self.GetFromConnections(conn, "delay")
                         for conn in self.get_connections(pop)]).flatten()

    def get_receptors(self, pop=None):
        return np.array([self.GetFromConnections(conn, "receptor")
                         for conn in self.get_connections(pop)]).flatten()

    def get_node_weight(self, pop=None):
        return unique(np.around(self.get_weights(pop), decimals=3)).tolist()

    def get_node_delay(self, pop=None):
        return unique(np.around(self.get_delays(pop), decimals=3)).tolist()

    def get_node_receptors(self, pop=None):
        return unique(self.get_receptors(pop)).tolist()  # pd.unique is faster than np.unique
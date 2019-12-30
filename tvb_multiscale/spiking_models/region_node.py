# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import flatten_tuple


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
        return flatten_tuple(super(SpikingRegionNode, self).__getitem__(keys))

    @abstractmethod
    def Get(self, params=None, indices_or_keys=None):
        pass

    @abstractmethod
    def Set(self, values_dict, indices_or_keys=None):
        pass

    @abstractmethod
    def _get_connections(self, neuron):
        pass

    @property
    def node(self):
        return self

    @property
    def populations(self):
        return list(self.index)

    def neurons(self, indices_or_keys=None):
        if indices_or_keys is None:
            return flatten_tuple(self)
        else:
            return self.__getitem__(indices_or_keys)

    @property
    def connections(self, indices_or_keys=None):
        connections = []
        for neuron in self.neurons(indices_or_keys):
            connections.append(self._get_connections(neuron))
        return tuple(connections)

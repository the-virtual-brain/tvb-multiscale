# -*- coding: utf-8 -*-

from collections import OrderedDict
from pandas import Series

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import flatten_tuple


LOG = initialize_logger(__name__)


class NESTRegionNode(Series):
    # This is an indexed mapping between populations labels and the
    # indices of neurons belonging to the corresponding populations
    nest_instance = None
    label = ""

    def __init__(self, nest_instance, label="", input_node=Series()):
        self.nest_instance = nest_instance
        self.label = str(label)
        super(NESTRegionNode, self).__init__(input_node)

    def __getitem__(self, keys):
        return flatten_tuple(super(NESTRegionNode, self).__getitem__(keys))

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

    def GetStatus(self, params=None, indices_or_keys=None):
        return self.nest_instance.GetStatus(self.neurons(indices_or_keys), params)

    def SetStatus(self, values_dict, indices_or_keys=None):
        self.nest_instance.SetStatus(self.neurons(indices_or_keys), values_dict)

    @property
    def connections(self, indices_or_keys=None):
        connections = []
        for neuron in self.neurons(indices_or_keys):
            connections.append(self.nest_instance.GetConnections(neuron))
        return tuple(connections)

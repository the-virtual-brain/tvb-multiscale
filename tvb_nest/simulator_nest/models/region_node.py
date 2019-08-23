# -*- coding: utf-8 -*-

from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import flatten_tuple
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict, OrderedDict

LOG = initialize_logger(__name__)


class NESTRegionNode(IndexedOrderedDict):
    nest_instance = None
    label = ""

    def __init__(self, nest_instance, node_ordered_dict=OrderedDict({}), label=""):
        super(NESTRegionNode, self).__init__(node_ordered_dict)
        self.nest_instance = nest_instance
        self.label = str(label)

    def __getitem__(self, keys):
        return flatten_tuple(super(NESTRegionNode, self).__getitem__(keys))

    @property
    def node(self):
        return self._dict

    @property
    def populations(self):
        return self.node.keys()

    def neurons(self, indices_or_keys=None):
        if indices_or_keys is None:
            return flatten_tuple(self.node.values())
        else:
            return self.__getitem__(indices_or_keys)

    def GetStatus(self, params=None, indices_or_keys=None):
        return self.nest_instance.GetStatus(self.neurons(indices_or_keys), params)

    def SetStatus(self, values_dict, indices_or_keys=None):
        self.nest_instance.set_values(self.neurons(indices_or_keys), values_dict)

    @property
    def connections(self, indices_or_keys=None):
        connections = []
        for neuron in self.neurons(indices_or_keys):
            connections.append(self.nest_instance.GetConnections(neuron))
        return tuple(connections)

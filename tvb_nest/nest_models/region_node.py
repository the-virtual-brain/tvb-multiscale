# -*- coding: utf-8 -*-

from pandas import Series

from tvb_multiscale.spiking_models.region_node import SpikingRegionNode


class NESTRegionNode(SpikingRegionNode):

    nest_instance = None

    def __init__(self, nest_instance, label="", input_node=Series()):
        self.nest_instance = nest_instance
        super(NESTRegionNode, self).__init__(label, input_node)

    def Get(self, params=None, indices_or_keys=None):
        return self.nest_instance.GetStatus(self.neurons(indices_or_keys), params)

    def Set(self, values_dict, indices_or_keys=None):
        self.nest_instance.SetStatus(self.neurons(indices_or_keys), values_dict)

    def _get_connections(self, neuron):
        return self.nest_instance.GetConnections((neuron, ))

    def GetFromConnections(self, connections, attr=None):
        if attr is None:
            return self.nest_instance.GetStatus(connections)[0]
        else:
            return self.nest_instance.GetStatus(connections, attr)[0]

    def SetToConnections(self, connections, values_dict):
        self.nest_instance.SetStatus(connections, values_dict)

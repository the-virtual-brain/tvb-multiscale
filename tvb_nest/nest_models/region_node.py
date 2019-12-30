# -*- coding: utf-8 -*-

from pandas import Series

from tvb_multiscale.spiking_models.region_node import SpikingRegionNode


class NESTRegionNode(SpikingRegionNode):

    def __init__(self, nest_instance, label="", input_node=Series()):
        self.nest_instance = nest_instance
        super(NESTRegionNode, self).__init__(label, input_node)

    def Get(self, params=None, indices_or_keys=None):
        return self.nest_instance.GetStatus(self.neurons(indices_or_keys), params)

    def Set(self, values_dict, indices_or_keys=None):
        self.nest_instance.SetStatus(self.neurons(indices_or_keys), values_dict)

    def _get_connections(self, neuron):
        return self.nest_instance.GetConnections(neuron)

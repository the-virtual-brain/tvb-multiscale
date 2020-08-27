# -*- coding: utf-8 -*-
from pandas import Series

from tvb_multiscale.spiking_models.region_node import SpikingRegionNode


class NESTRegionNode(SpikingRegionNode):

    nest_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, label="", input_node=None, nest_instance=None, **kwargs):
        self.nest_instance = nest_instance
        super(SpikingRegionNode, self).__init__(label, input_node, **kwargs)

    @property
    def spiking_simulator_module(self):
        if self.nest_instance is None:
            for i_pop, pop_lbl, pop in self._loop_generator():
                self.nest_instance = pop.nest_instance
                if self.nest_instance is not None:
                    break
        return self.nest_instance
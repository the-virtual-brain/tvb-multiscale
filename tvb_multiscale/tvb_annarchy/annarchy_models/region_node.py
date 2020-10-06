# -*- coding: utf-8 -*-
from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode


class ANNarchyRegionNode(SpikingRegionNode):

    annarchy_instance = None
    _weight_attr = "weights"
    _delay_attr = "delays"
    _receptor_attr = "target"

    def __init__(self, label="", input_nodes=None, annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        super(ANNarchyRegionNode, self).__init__(label, input_nodes, **kwargs)

    @property
    def spiking_simulator_module(self):
        if self.annarchy_instance is None:
            for i_pop, pop_lbl, pop in self._loop_generator():
                self.annarchy_instance = pop.annarchy_instance
                if self.annarchy_instance is not None:
                    break
        return self.annarchy_instance
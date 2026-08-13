# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode


class ANNarchyRegionNode(SpikingRegionNode):

    """ANNarchyRegionNode class is an indexed mapping
       (based on inheriting from pandas.Series class)
       between populations labels and ANNarchy.Population instances,
       residing at a specific brain region node.
    """

    _weight_attr = "weights"
    _delay_attr = "delays"
    _receptor_attr = "target"

    @property
    def annarchy_instance(self):
        return self.spiking_simulator_module

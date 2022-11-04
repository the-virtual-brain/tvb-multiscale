# -*- coding: utf-8 -*-
from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode


class NESTRegionNode(SpikingRegionNode):

    """NESTRegionNode class is an indexed mapping
       (based on inheriting from pandas.Series class)
       between populations labels and nest.NodeCollection instances,
       residing at a specific brain region node.
    """

    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    @property
    def nest_instance(self):
        return self.spiking_simulator_module

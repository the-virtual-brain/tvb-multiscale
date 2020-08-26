# -*- coding: utf-8 -*-

from tvb_multiscale.spiking_models.region_node import SpikingRegionNode


class NESTRegionNode(SpikingRegionNode):

    nest_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor_type"

    @property
    def spiking_simulator_module(self):
        return self.nest_instance
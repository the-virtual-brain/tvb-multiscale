# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.brain import SpikingBrain


class ANNarchyBrain(SpikingBrain):

    """ANNarchyBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective ANNarchyRegionNode instances.
    """

    _weight_attr = "weights"
    _delay_attr = "delays"
    _receptor_attr = "target"

    @property
    def annarchy_instance(self):
        return self.spiking_simulator_module

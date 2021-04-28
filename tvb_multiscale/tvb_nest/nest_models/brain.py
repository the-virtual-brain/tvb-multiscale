# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.brain import SpikingBrain


class NESTBrain(SpikingBrain):

    """"NESTBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective NESTRegionNode instances.
    """

    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    @property
    def nest_instance(self):
        return self.spiking_simulator_module

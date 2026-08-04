# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.brain import SpikingBrain


# TODO: why is this ever needed?

class NetpyneBrain(SpikingBrain):

    """"NetpyneBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective NetpyneRegionNode instances.
    """

    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    @property
    def netpyne_instance(self):
        return self.spiking_simulator_module

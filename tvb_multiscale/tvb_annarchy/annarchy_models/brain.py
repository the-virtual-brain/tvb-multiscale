# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.brain import SpikingBrain


class ANNarchyBrain(SpikingBrain):

    """ANNarchyBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective ANNarchyRegionNode instances.
    """

    annarchy_instance = None
    _weight_attr = "weights"
    _delay_attr = "delays"
    _receptor_attr = "target"

    def __init__(self, input_brain=None, annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        super(ANNarchyBrain, self).__init__(input_brain, **kwargs)

    @property
    def spiking_simulator_module(self):
        if self.annarchy_instance is None:
            for i_pop, pop_lbl, pop in self._loop_generator():
                self.annarchy_instance = pop.annarchy_instance
                if self.annarchy_instance is not None:
                    break
        return self.annarchy_instance

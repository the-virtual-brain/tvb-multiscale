# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import load_annarchy

from tvb_multiscale.core.spiking_models.network import SpikingNetwork


LOG = initialize_logger(__name__)


class ANNarchyNetwork(SpikingNetwork):

    annarchy_instance = None

    _dt = None

    def __init__(self, annarchy_instance=None,
                 brain_regions=None,
                 output_devices=None,
                 input_devices=None,
                 config=CONFIGURED):
        if annarchy_instance is None:
            annarchy_instance = load_annarchy(self.config, LOG)
        self.annarchy_instance = annarchy_instance
        super(ANNarchyNetwork, self).__init__(brain_regions, output_devices, input_devices, config)

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @property
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

    @property
    def min_delay(self):
        return self.dt

    def configure(self, *args, **kwargs):
        # Run last configurations before simulation. Maybe compile()?
        pass

    def Run(self, *args, **kwargs):
        measure_time = kwargs.pop("measure_time", True)
        raise self.annarchy_instance.simulate(1000.0, measure_time=measure_time, **kwargs)

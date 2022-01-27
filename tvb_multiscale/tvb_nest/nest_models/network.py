# -*- coding: utf-8 -*-

import pandas as pd

from tvb_multiscale.tvb_nest.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
from tvb_multiscale.tvb_nest.nest_models.devices import NESTOutputSpikeDeviceDict, NESTOutputContinuousTimeDeviceDict
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


LOG = initialize_logger(__name__)


class NESTNetwork(SpikingNetwork):
    """
        NESTNetwork is a class representing a NEST spiking network comprising of:
        - a NESTBrain class, i.e., neural populations organized per brain region they reside and neural model,
        - a pandas.Series of DeviceSet classes of output (measuring/recording/monitor) NEST devices,
        - a pandas.Series of DeviceSet classes of input (stimulating) NEST devices,
        all of which are implemented as indexed mappings by inheriting from pandas.Series class.
        The class also includes methods to return measurements (mean, sum/total data, spikes, spikes rates etc)
        from output devices, as xarray.DataArrays.
        e.g. NESTPopulations can be indexed as:
        nest_network.brain_regions['rh-insula']['E'] for population "E" residing in region node "rh-insula",
        and similarly for an output device:
        nest_network.output_devices['Excitatory']['rh-insula'],
        which measures a quantity labelled following the target population ("Excitatory"),
        residing in region node "rh-insula".
    """

    nest_instance = None

    _OutputSpikeDeviceDict = NESTOutputSpikeDeviceDict
    _OutputContinuousTimeDeviceDict = NESTOutputContinuousTimeDeviceDict

    def __init__(self, nest_instance=None,
                 brain_regions=pd.Series(),
                 output_devices=pd.Series(),
                 input_devices=pd.Series(),
                 config=CONFIGURED):
        if not isinstance(brain_regions, NESTBrain):
            brain_regions = NESTBrain(brain_regions)
        self.nest_instance = nest_instance
        super(NESTNetwork, self).__init__(brain_regions, output_devices, input_devices, config)

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    def Run(self, time):
        if self.nest_instance is not None:
            self.nest_instance.Run(time)

    def Simulate(self, time):
        if self.nest_instance is not None:
            self.nest_instance.Prepare()
            self.nest_instance.Run(time)

    def Cleanup(self):
        if self.nest_instance is not None:
            self.nest_instance.Cleanup()

    @property
    def min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def dt(self):
        return self.nest_instance.GetKernelStatus("resolution")

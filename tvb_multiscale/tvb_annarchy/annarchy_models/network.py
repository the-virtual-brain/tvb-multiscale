# -*- coding: utf-8 -*-

import os

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import load_annarchy
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyOutputSpikeDeviceDict, ANNarchyOutputContinuousTimeDeviceDict

from tvb_multiscale.core.spiking_models.network import SpikingNetwork


LOG = initialize_logger(__name__)


class ANNarchyNetwork(SpikingNetwork):

    """
        ANNarchyNetwork is a class representing a ANNarchy spiking network comprising of:
        - a ANNarchyBrain class, i.e., neural populations organized per brain region they reside and neural model,
        - a pandas.Series of DeviceSet classes of output (measuring/recording/monitor) devices,
        - a pandas.Series of DeviceSet classes of input (stimulating) devices,
        all of which are implemented as indexed mappings by inheriting from pandas.Series class.
        The class also includes methods to return measurements (mean, sum/total data, spikes, spikes rates etc)
        from output devices, as xarray.DataArrays.
        e.g. ANnarchyPopulations can be indexed as:
        annarchy_network.brain_regions['rh-insula']['E'] for population "E" residing in region node "rh-insula",
        and similarly for an output device:
        annarchy_network.output_devices['Excitatory']['rh-insula'],
        which measures a quantity labelled following the target population ("Excitatory"),
        residing in region node "rh-insula".
    """

    annarchy_instance = None

    _network = None

    _dt = None

    _OutputSpikeDeviceDict = ANNarchyOutputSpikeDeviceDict
    _OutputContinuousTimeDeviceDict = ANNarchyOutputContinuousTimeDeviceDict

    def __init__(self, annarchy_instance=None,
                 brain_regions=None,
                 output_devices=None,
                 input_devices=None,
                 config=CONFIGURED):
        if annarchy_instance is None:
            annarchy_instance = load_annarchy(self.config, LOG)
        self.annarchy_instance = annarchy_instance
        super(ANNarchyNetwork, self).__init__(brain_regions, output_devices, input_devices, config)
        self._OutputSpikeDeviceDict = ANNarchyOutputSpikeDeviceDict
        self._OutputContinuousTimeDeviceDict = ANNarchyOutputContinuousTimeDeviceDict

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @property
    def network(self):
        if self._network is None:
            self._network = self.annarchy_instance.Network(everything=True)
        return self._network

    @property
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

    @property
    def min_delay(self):
        return self.dt

    def Run(self, simulation_length, **kwargs):
        """Method to simulate the ANNarchy network for a specific simulation_length (in ms).
           It will run annarchy_instance.simulate(simulation_length, *args, **kwargs)
        """
        measure_time = kwargs.pop("measure_time", False)
        for dev_name, out_dev_set in self.output_devices.iteritems():
            out_dev_set.do_for_all_devices("resume")
        self.annarchy_instance.simulate(simulation_length, measure_time=measure_time, **kwargs)
        for dev_name, out_dev_set in self.output_devices.iteritems():
            out_dev_set.do_for_all_devices("pause")

# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDeviceDict, ANNarchyCurrentInputDeviceDict, ANNarchySpikeInputDeviceDict,\
    ANNarchyOutputDeviceDict, ANNarchyOutputContinuousTimeDeviceDict, ANNarchyOutputSpikeDeviceDict

from tvb_multiscale.core.interfaces.base import TVBSpikeNetInterface


class TVBANNarchyInterface(TVBSpikeNetInterface):
    _available_input_devices = ANNarchyInputDeviceDict.keys()
    _current_input_devices = ANNarchyCurrentInputDeviceDict.keys()
    _spike_rate_input_devices = ANNarchySpikeInputDeviceDict.keys()
    _available_output_devices = ANNarchyOutputDeviceDict.keys()
    _spike_rate_output_devices = ANNarchyOutputSpikeDeviceDict.keys()
    _multimeter_output_devices = ANNarchyOutputContinuousTimeDeviceDict.keys()

    def __init__(self, config=CONFIGURED):
        super(TVBANNarchyInterface, self).__init__(config)

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

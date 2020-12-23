# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.devices import \
    NESTInputDeviceDict, NESTSpikeInputDeviceDict, NESTCurrentInputDeviceDict, \
    NESTOutputDeviceDict, NESTOutputSpikeDeviceDict, NESTOutputContinuousTimeDeviceDict
from tvb_multiscale.core.interfaces.base import TVBSpikeNetInterface


class TVBNESTInterface(TVBSpikeNetInterface):
    _available_input_devices = NESTInputDeviceDict.keys()
    _current_input_devices = NESTCurrentInputDeviceDict.keys()
    _spike_rate_input_devices = NESTSpikeInputDeviceDict.keys()
    _available_output_devices = NESTOutputDeviceDict.keys()
    _spike_rate_output_devices = NESTOutputSpikeDeviceDict.keys()
    _multimeter_output_devices = NESTOutputContinuousTimeDeviceDict.keys()
    _voltmeter_output_devices = ["voltmeter"]

    def __init__(self, config=CONFIGURED):
        super(TVBNESTInterface, self).__init__(config)

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

# -*- coding: utf-8 -*-

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.devices import NESTInputDeviceDict, NESTOutputDeviceDict, NESTOutputSpikeDeviceDict
from tvb_multiscale.interfaces.base import TVBSpikeNetInterface


class TVBNESTInterface(TVBSpikeNetInterface):
    _available_input_devices = NESTInputDeviceDict.keys()
    _current_input_devices = ["dc_generator"]
    _spike_rate_input_devices = ["poisson_generator", "spike_generator",
                                 "mip_generator", "inhomogeneous_poisson_generator"]
    _available_output_devices = NESTOutputDeviceDict.keys()
    _spike_rate_output_devices = NESTOutputSpikeDeviceDict.keys()
    _multimeter_output_devices = ["multimeter"]
    _voltmeter_output_devices = ["voltmeter"]

    def __init__(self, config=CONFIGURED):
        super(TVBNESTInterface, self).__init__(config)

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

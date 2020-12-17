# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.tvb_to_nest_devices_interface import INPUT_INTERFACES_DICT
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device
from tvb_multiscale.core.interfaces.builders.tvb_to_spikeNet_device_interface_builder import \
    TVBtoSpikeNetDeviceInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices


class TVBtoNESTDeviceInterfaceBuilder(TVBtoSpikeNetDeviceInterfaceBuilder):
    _available_input_device_interfaces = INPUT_INTERFACES_DICT

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    @property
    def spiking_dt(self):
        try:
            return self.nest_instance.GetKernelStatus("resolution")
        except:
            return super(TVBtoNESTDeviceInterfaceBuilder, self).spiking_dt

    @property
    def min_delay(self):
        try:
            return self.nest_instance.GetKernelStatus("min_delay")
        except:
            return self.default_min_delay

    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, nest_instance=self.nest_instance)

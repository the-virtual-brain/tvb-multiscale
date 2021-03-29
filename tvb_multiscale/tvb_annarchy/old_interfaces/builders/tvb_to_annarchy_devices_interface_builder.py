# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.old_interfaces.tvb_to_annarchy_devices_interface import INPUT_INTERFACES_DICT
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import create_device, connect_device

from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBtoSpikeNetDeviceInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices


class TVBtoANNarchyDeviceInterfaceBuilder(TVBtoSpikeNetDeviceInterfaceBuilder):
    _available_input_device_interfaces = INPUT_INTERFACES_DICT

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

    @property
    def spiking_dt(self):
        try:
            return self.annarchy_instance.dt()
        except:
            return super(TVBtoANNarchyDeviceInterfaceBuilder, self).spiking_dt

    @property
    def min_delay(self):
        return self.spiking_dt

    def build_and_connect_devices(self, devices, nodes):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, config=self.config, devices=self.spiking_network.input_proxies,
                                         annarchy_instance=self.annarchy_instance)

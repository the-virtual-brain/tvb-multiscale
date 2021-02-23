# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.nest_to_tvb_interface import NESTtoTVBinterface
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import create_device, connect_device
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.core.interfaces.builders.spikeNet_to_tvb_interface_builder import SpikeNetToTVBInterfaceBuilder


class NESTtoTVBInterfaceBuilder(SpikeNetToTVBInterfaceBuilder):
    _build_target_class = NESTtoTVBinterface

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    def build_and_connect_devices(self, devices, nodes):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, devices=self.spiking_network.output_proxies,
                                         nest_instance=self.nest_instance)

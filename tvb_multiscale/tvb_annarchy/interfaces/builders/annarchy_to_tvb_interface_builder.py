# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.interfaces.annarchy_to_tvb_interface import ANNarchytoTVBinterface
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import create_device, connect_device

from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.core.interfaces.builders.spikeNet_to_tvb_interface_builder import SpikeNetToTVBInterfaceBuilder


class ANNarchytoTVBInterfaceBuilder(SpikeNetToTVBInterfaceBuilder):
    _build_target_class = ANNarchytoTVBinterface

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, annarchy_instance=self.annarchy_instance)

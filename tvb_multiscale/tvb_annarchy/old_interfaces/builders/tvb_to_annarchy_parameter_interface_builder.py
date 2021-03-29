# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.old_interfaces.tvb_to_annarchy_parameters_interface import TVBtoANNarchyParameterInterface

from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBtoSpikeNetParameterInterfaceBuilder


class TVBtoANNarchyParameterInterfaceBuilder(TVBtoSpikeNetParameterInterfaceBuilder):
    _build_target_class = TVBtoANNarchyParameterInterface

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

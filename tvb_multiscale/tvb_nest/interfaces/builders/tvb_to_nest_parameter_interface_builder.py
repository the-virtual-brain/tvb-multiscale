# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.tvb_to_nest_parameters_interfaces import TVBtoNESTParameterInterface
from tvb_multiscale.core.interfaces.builders.tvb_to_spikeNet_parameter_interface_builder import \
    TVBtoSpikeNetParameterInterfaceBuilder


class TVBtoNESTParameterInterfaceBuilder(TVBtoSpikeNetParameterInterfaceBuilder):
    _build_target_class = TVBtoNESTParameterInterface

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

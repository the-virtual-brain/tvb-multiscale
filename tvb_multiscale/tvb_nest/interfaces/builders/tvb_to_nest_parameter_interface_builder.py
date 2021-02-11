# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.tvb_to_nest_parameters_interface import TVBtoNESTParameterInterface
from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBtoSpikeNetParameterInterfaceBuilder


class TVBtoNESTParameterInterfaceBuilder(TVBtoSpikeNetParameterInterfaceBuilder):
    _build_target_class = TVBtoNESTParameterInterface

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance
